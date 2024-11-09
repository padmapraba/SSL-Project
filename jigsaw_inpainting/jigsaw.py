
import torch.nn as nn
import torch
import cv2
import torch
import random
import itertools  # Add this import to fix the NameError
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import STL10
from torchvision.models import resnet18


class JigsawSTL10Dataset(Dataset):
    def __init__(self, dataset, grid_size=3):
        self.dataset = dataset
        self.grid_size = grid_size

    def __getitem__(self, index):
        img, _ = self.dataset[index]
        shuffled_patches, perm_class = create_jigsaw_puzzle(img, self.grid_size)
        return shuffled_patches, perm_class

    def __len__(self):
        return len(self.dataset)

# Predefine 1000 fixed permutations from the 9! possible permutations for the jigsaw task
def create_permutations(num_patches=9, num_permutations=1000):
    all_permutations = list(itertools.permutations(range(num_patches)))
    random.seed(42)  # For reproducibility
    selected_permutations = random.sample(all_permutations, num_permutations)
    return selected_permutations

permutations = create_permutations()

def create_jigsaw_puzzle(image, grid_size=3, permutations=permutations):
    _, height, width = image.shape
    patch_h, patch_w = height // grid_size, width // grid_size
    patches = []

    # Extract the patches from the image
    for i in range(grid_size):
        for j in range(grid_size):
            patch = image[:, i * patch_h: (i + 1) * patch_h, j * patch_w: (j + 1) * patch_w]
            patches.append(patch)

    # Select a random permutation index from the predefined set
    perm_class = random.choice(range(len(permutations)))
    perm = permutations[perm_class]

    # Shuffle the patches based on the selected permutation
    shuffled_patches = [patches[i] for i in perm]
    
    return torch.stack(shuffled_patches), torch.tensor(perm_class, dtype=torch.long)


# ResNet-based Jigsaw Puzzle Solver
class JigsawResNet(nn.Module):
    def __init__(self, num_patches, num_permutations):
        super(JigsawResNet, self).__init__()
        # ResNet-18 backbone
        self.backbone = resnet18(weights=None)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])  # Remove last fully connected layer

        # Fully connected layer for jigsaw permutation classification
        self.fc = nn.Sequential(
            nn.Linear(512 * num_patches, 1024),  # 512 is output feature size from ResNet, num_patches = 9
            nn.ReLU(),
            nn.Linear(1024, num_permutations)
        )

    def forward(self, x):
        # x.shape = [batch_size, num_patches, channels, height, width]
        batch_size, num_patches, channels, height, width = x.shape
        patches = []

        # Process each patch independently
        for i in range(num_patches):
            patch_features = self.backbone(x[:, i])  # Extract features for each patch
            patch_features = torch.flatten(patch_features, start_dim=1)  # Flatten each patch's features
            patches.append(patch_features)
            # patch = x[:, i]  # Extract the i-th patch: [batch_size, channels, height, width]
            # patch_features = self.backbone(patch)  # Pass through backbone
            
            # patch_features = patch_features.view(batch_size, -1)  # Flatten to [batch_size, 512]
            # patches.append(patch_features)

        # Concatenate features from all patches
        concatenated_features = torch.cat(patches, dim=1)  # Shape: [batch_size, 512 * num_patches]
        
        # Pass concatenated features through fully connected layers
        output = self.fc(concatenated_features)
        return output
