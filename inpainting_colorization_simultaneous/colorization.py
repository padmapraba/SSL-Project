import torch.nn as nn
import torch
import cv2
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import STL10

class RGB2LabTransform:
    def __call__(self, image):
        # Convert PyTorch tensor (C x H x W) to NumPy array (H x W x C)
        np_image = image.numpy().transpose(1, 2, 0)
        
        # Convert RGB to LAB using OpenCV
        lab_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2LAB)
        
        # Extract L channel and AB channels
        L_channel = lab_image[:, :, 0] / 255.0  # Normalize L channel to [0, 1]
        AB_channels = (lab_image[:, :, 1:] + 128) / 255.0  # Normalize AB channels to [0, 1]
        
        # Convert L and AB channels back to PyTorch tensors
        L_tensor = torch.tensor(L_channel).unsqueeze(0)  # Add channel dimension for L
        AB_tensor = torch.tensor(AB_channels).permute(2, 0, 1)  # Permute AB channels to (C, H, W)
        
        return L_tensor, AB_tensor


class STL10ColorizationDataset(STL10):
    def __getitem__(self, index):
        # Get the original image from the parent class
        image, _ = super().__getitem__(index)
        
        # Apply the LAB transformation to the image
        L_channel, AB_channels = RGB2LabTransform()(image)
        
        return L_channel, AB_channels

# Colorizarion class
class Colorization(nn.Module):
    def __init__(self, backbone):
        super(Colorization, self).__init__()
        self.backbone = backbone

        self.colorization_head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 2, kernel_size=3, padding=1),
            nn.Tanh()  # Output should be in range [-1, 1]
        )
        
        # Upsampling to 96x96
        self.upsample = nn.Upsample(size=(96, 96), mode='bilinear', align_corners=False)
        
    def forward(self, L_channel):
        features = self.backbone(L_channel)

        AB_channels = self.colorization_head(features)
        AB_channels = self.upsample(AB_channels)
        return AB_channels


class EarlyStopping:
    def __init__(self, paitence=15, min_delta=0):
        self.paitence = paitence
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.paitence:
                self.early_stop = True