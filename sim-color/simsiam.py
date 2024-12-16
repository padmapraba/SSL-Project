import math

import numpy as np
import torch
import torch.nn as nn
import torchvision

from lightly.data import LightlyDataset
from lightly.loss import NegativeCosineSimilarity
from lightly.models.modules.heads import SimSiamPredictionHead, SimSiamProjectionHead
from lightly.transforms import SimCLRTransform, utils
import torch
import torchvision
from torchvision.datasets import STL10
from lightly.data import LightlyDataset




# Set parameters
num_workers = 8
batch_size = 256 
seed = 1
epochs = 100
input_size = 96  

# Embedding and prediction parameters
num_ftrs = 512
out_dim = proj_hidden_dim = 512
pred_hidden_dim = 128


# Define the augmentations for self-supervised learning
transform = SimCLRTransform(
    input_size=input_size,
    hf_prob=0.5,
    vf_prob=0.5,
    rr_prob=0.5,
    min_scale=0.5,
    cj_prob=0.2,
    cj_bright=0.1,
    cj_contrast=0.1,
    cj_hue=0.1,
    cj_sat=0.1,
)

# Load the STL10 training data
stl10_train = STL10(root='../data', split='unlabeled', download=True, transform=transform)
dataset_train_simsiam = LightlyDataset.from_torch_dataset(stl10_train)

# Dataloader for training
dataloader_train_simsiam = torch.utils.data.DataLoader(
    dataset_train_simsiam,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers,
)

# Define transformations for testing and embedding
test_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize((input_size, input_size)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],  # STL10 dataset mean values
            std=[0.247, 0.243, 0.261],  # STL10 dataset std values
        ),
    ]
)

# Load STL10 test dataset
stl10_test = STL10(root='../data', split='test', download=True, transform=test_transforms)
dataset_test = LightlyDataset.from_torch_dataset(stl10_test)

# Dataloader for testing/embedding
dataloader_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers,
)



# Create SimSiam model
class SimSiam(nn.Module):
    def __init__(self, backbone, num_ftrs, proj_hidden_dim, pred_hidden_dim, out_dim):
        super().__init__()
        self.backbone = backbone
        self.projection_head = SimSiamProjectionHead(num_ftrs, proj_hidden_dim, out_dim)
        self.prediction_head = SimSiamPredictionHead(out_dim, pred_hidden_dim, out_dim)

    def forward(self, x):
        # get representations
        f = self.backbone(x).flatten(start_dim=1)
        # get projections
        z = self.projection_head(f)
        # get predictions
        p = self.prediction_head(z)
        # stop gradient
        z = z.detach()
        return z, p


# we use a pretrained resnet for this tutorial to speed
# up training time but you can also train one from scratch
resnet = torchvision.models.resnet18()
backbone = nn.Sequential(*list(resnet.children())[:-1])
model = SimSiam(backbone, num_ftrs, proj_hidden_dim, pred_hidden_dim, out_dim)

# SimSiam uses a symmetric negative cosine similarity loss
criterion = NegativeCosineSimilarity()

# scale the learning rate
lr = 0.01 * batch_size / 256
# use SGD with momentum and weight decay
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)



#Train SimSiam
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

avg_loss = 0.0
avg_output_std = 0.0
for e in range(epochs):
    for (x0, x1), _, _ in dataloader_train_simsiam:
        # move images to the gpu
        x0 = x0.to(device)
        x1 = x1.to(device)

        # run the model on both transforms of the images
        # we get projections (z0 and z1) and
        # predictions (p0 and p1) as output
        z0, p0 = model(x0)
        z1, p1 = model(x1)

        # apply the symmetric negative cosine similarity
        # and run backpropagation
        loss = 0.5 * (criterion(z0, p1) + criterion(z1, p0))
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        # calculate the per-dimension standard deviation of the outputs
        # we can use this later to check whether the embeddings are collapsing
        output = p0.detach()
        output = torch.nn.functional.normalize(output, dim=1)

        output_std = torch.std(output, 0)
        output_std = output_std.mean()

        # use moving averages to track the loss and standard deviation
        w = 0.9
        avg_loss = w * avg_loss + (1 - w) * loss.item()
        avg_output_std = w * avg_output_std + (1 - w) * output_std.item()

    # the level of collapse is large if the standard deviation of the l2
    # normalized output is much smaller than 1 / sqrt(dim)
    collapse_level = max(0.0, 1 - math.sqrt(out_dim) * avg_output_std)
    # print intermediate results
    print(
        f"[Epoch {e:3d}] "
        f"Loss = {avg_loss:.2f} | "
        f"Collapse Level: {collapse_level:.2f} / 1.00"
    )
    
    
    import matplotlib.pyplot as plt

# Data for the first segment of epochs (Epochs 0–42)
epoch1 = list(range(50))
loss1 = [
    -0.85, -0.88, -0.87, -0.87, -0.86, -0.85, -0.88, -0.88, -0.90, -0.88, 
    -0.90, -0.89, -0.88, -0.88, -0.89, -0.89, -0.89, -0.89, -0.90, -0.89,
    -0.89, -0.89, -0.90, -0.90, -0.90, -0.90, -0.90, -0.90, -0.89, -0.90, 
    -0.90, -0.90, -0.90, -0.90, -0.90, -0.90, -0.90, -0.91, -0.91, -0.90,
    -0.90, -0.91, -0.91, -0.91, -0.91, -0.92, -0.91, -0.92, -0.91, -0.91
]
plt.figure(figsize=(12, 6))

# First loss plot (Epochs 0–42)
plt.plot(epoch1, loss1, label="Loss Segment 1 (Epochs 0-42)", color="blue")


new_backbone  = nn.Sequential(*list(model.backbone.children())[:-1])
torch.save(new_backbone.state_dict(), 'backbone_weights_6.pth')
