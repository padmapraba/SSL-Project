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
from colorization import Colorization

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

# First load the pretrained colorization model
resnet = torchvision.models.resnet18()
colorization_backbone = nn.Sequential(*list(resnet.children())[:-1])
colorization_model = Colorization(colorization_backbone)
colorization_model.load_state_dict(torch.load("colorization_model_weights_final.pth"))

# Use the trained colorization backbone for MoCo
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

# Use the pretrained colorization backbone
model = MoCo(colorization_model.backbone)

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
        query = model(x_query)
        key = model.forward_momentum(x_key)
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
torch.save(new_backbone.state_dict(), 'models/backbone_weights_200.pth')

# After training is complete, create the plot
import matplotlib.pyplot as plt
plt.style.use('seaborn')
plt.figure(figsize=(10, 6))
plt.plot(losses, color='blue', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('MoCo Training Loss', fontsize=14, pad=15)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('training_loss_plot.png', dpi=300, bbox_inches='tight')
plt.close()
