import copy
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision

from lightly.data import LightlyDataset
from lightly.loss import NTXentLoss
from lightly.models import ResNetGenerator
from lightly.models.modules.heads import MoCoProjectionHead
from lightly.models.utils import (
    batch_shuffle,
    batch_unshuffle,
    deactivate_requires_grad,
    update_momentum,
)
from lightly.transforms import MoCoV2Transform, utils

from lightly.loss import NTXentLoss
from lightly.models.modules import MoCoProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.transforms.moco_transform import MoCoV2Transform
from lightly.utils.scheduler import cosine_schedule
from inpainting import InpaintingModel, Encoder




if torch.cuda.is_available():
    device = torch.device("cuda")  # Use NVIDIA GPU
    print('cuda')
elif torch.backends.mps.is_available():
    device = torch.device("mps")   # Use Apple's Metal (for M1/M2 Macs)
    print('mps')
else:
    device = torch.device("cpu") 
    print('cpu')
    
    
    
    
num_workers = 8
batch_size = 256  # STL-10 is larger, so a smaller batch size may be better for memory
memory_bank_size = 4096
seed = 1
max_epochs = 100

path_to_data = "../data"

import copy
import torch
import torchvision
from torch import nn
from torchvision.datasets import STL10

from lightly.loss import NTXentLoss
from lightly.models.modules import MoCoProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.transforms.moco_transform import MoCoV2Transform
from lightly.utils.scheduler import cosine_schedule






# First load the pretrained inpainting model
inpainting_encoder = Encoder()
inpainting_model = InpaintingModel(inpainting_encoder)
checkpoint = torch.load("inpainting_model_gen_weights_final.pth", map_location=device)
inpainting_model.load_state_dict(checkpoint["model_state_dict"])


# Use the trained inpainting backbone for MoCo
class MoCo(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        self.backbone = backbone
        self.projection_head = MoCoProjectionHead(512, 512, 128)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

    def forward(self, x):
        query = self.backbone(x).flatten(start_dim=1)
        query = self.projection_head(query)
        return query

    def forward_momentum(self, x):
        key = self.backbone_momentum(x).flatten(start_dim=1)
        key = self.projection_head_momentum(key).detach()
        return key

# Use the pretrained inpainting encoder
model = MoCo(inpainting_model.encoder)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)






transform = MoCoV2Transform(input_size=96)
dataset = torchvision.datasets.STL10(
    root='../data', split='train+unlabeled', download=True, transform=transform
)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=256,
    shuffle=True,
    drop_last=True,
    num_workers=8,
)

criterion = NTXentLoss(memory_bank_size=(4096, 128))
optimizer = torch.optim.SGD(model.parameters(), lr=0.06)





import json

epochs = 200
losses = []

print("Starting Training")
for epoch in range(epochs):
    total_loss = 0
    momentum_val = cosine_schedule(epoch, epochs, 0.996, 1)
    for batch in dataloader:
        x_query, x_key = batch[0]
        update_momentum(model.backbone, model.backbone_momentum, m=momentum_val)
        update_momentum(
            model.projection_head, model.projection_head_momentum, m=momentum_val
        )
        x_query = x_query.to(device)
        x_key = x_key.to(device)
        
        # Get encoder features and reshape
        query_features = model.backbone(x_query)
        query_features = torch.flatten(query_features, start_dim=2)
        query_features = query_features.mean(dim=2)  # Average pool spatial dimensions
        query = model.projection_head(query_features)
        
        # Get momentum encoder features and reshape
        key_features = model.backbone_momentum(x_key)
        key_features = torch.flatten(key_features, start_dim=2)
        key_features = key_features.mean(dim=2)  # Average pool spatial dimensions
        key = model.projection_head_momentum(key_features).detach()
        
        loss = criterion(query, key)
        total_loss += loss.detach()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    avg_loss = total_loss / len(dataloader)
    losses.append(avg_loss.item())
    print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")
    
    # Save training progress every 10 epochs
    if (epoch + 1) % 10 == 0:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'losses': losses
        }
        torch.save(checkpoint, f'models/checkpoint_epoch_{epoch+1}.pth')
        
        # Save losses to JSON
        with open('training_losses.json', 'w') as f:
            json.dump({'losses': losses}, f)

new_backbone = nn.Sequential(*list(model.backbone.children())[:-1])
torch.save(new_backbone.state_dict(), 'models/backbone_weights_fin.pth')










new_backbone = nn.Sequential(*list(model.backbone.children())[:-1])
torch.save(new_backbone.state_dict(), 'models/colorization_model_weights_final.pth')

# class ClassificationNet(nn.Module):
#     def __init__(self, backbone, num_classes):
#         super(ClassificationNet, self).__init__()
#         self.backbone = backbone
#         self.classifier = nn.Linear(512, num_classes)

#     def forward(self, x):
#         features = self.backbone(x)
#         pooled_features = nn.AdaptiveAvgPool2d((1, 1))(features)
#         pooled_features = pooled_features.view(pooled_features.size(0), -1)
#         output = self.classifier(pooled_features)
#         return output

# classification_model = ClassificationNet(new_backbone, num_classes=10).to(device)

class ClassificationNet(nn.Module):
    def __init__(self, backbone, num_classes):
        super(ClassificationNet, self).__init__()
        
        # Keep only the necessary adaptation layer to match backbone's expected input
        self.conv_adapt = nn.Conv2d(3, 512, kernel_size=3, padding=1)
        
        self.backbone = backbone
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_adapt(x)  # Adapt the input channels
        x = self.backbone(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

# Create the classification model
classification_model = ClassificationNet(new_backbone, num_classes=10).to(device)


# Update the transforms to ensure correct input formatting
classification_transform = transforms.Compose([
    transforms.Resize((96, 96)),  # Ensure input size matches what the backbone expects
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import STL10
stl10_train = STL10(root='../data', split='train', download=True, transform=classification_transform)
stl10_test = STL10(root='../data', split='test', download=True, transform=classification_transform)

# Fine-tuning: Load training data for classification task
train_loader = DataLoader(stl10_train, batch_size=64, shuffle=True)

# Testing: Load test data for final evaluation
test_loader = DataLoader(stl10_test, batch_size=64, shuffle=True)

import os
import json
import torch.nn as nn
import torch.optim as optim

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()  # multi-class classification
optimizer = optim.Adam(classification_model.parameters(), lr=1e-4)

# Ensure directory for saving models and logs exists
save_dir = 'models/downstream'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Initialize JSON log file
log_file = os.path.join(save_dir, 'training_log.json')
training_logs = {"epoch_losses": []}

# Training Loop
num_epochs = 150
for epoch in range(num_epochs):
    classification_model.train()  
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = classification_model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

    # Save epoch loss to JSON
    training_logs["epoch_losses"].append({"epoch": epoch + 1, "loss": avg_loss})

    # Save model weights every 10 epochs
    if (epoch + 1) % 10 == 0:
        torch.save(
            classification_model.state_dict(),
            os.path.join(save_dir, f'classification_model_weights_epoch_{epoch+1}.pth')
        )

# Save final model weights
final_model_path = os.path.join(save_dir, 'classification_model_weights_final_200.pth')
torch.save(classification_model.state_dict(), final_model_path)

# Save the JSON logs
with open(log_file, 'w') as file:
    json.dump(training_logs, file, indent=4)

print(new_backbone)
