import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10  # This automatically handles the dataset splitting in memory
from torchvision.datasets import STL10
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np


# Added due to combining with colorization
def mask_image(img, mask_size):
    masked_img = img.clone()  # Clone to preserve original image
    batch_size, _, h, w = img.shape  # Unpack batch dimension as well

    # Loop over each image in the batch and apply a random mask
    for i in range(batch_size):
        y = torch.randint(0, h - mask_size, (1,)).item()
        x = torch.randint(0, w - mask_size, (1,)).item()
        masked_img[i, :, y:y+mask_size, x:x+mask_size] = 0  # Set mask area to zero

    # Create a mask tensor indicating where the masked area is
    masks = torch.zeros_like(img)
    for i in range(batch_size):
        masks[i, :, y:y+mask_size, x:x+mask_size] = 1

    return masked_img, masks

# def mask_image(img, mask_size):
#     masked_img = img.clone()  # clone to preserve original image
#     _, h, w = img.shape
#     y = torch.randint(0, h - mask_size, (1,)).item()
#     x = torch.randint(0, w - mask_size, (1,)).item()
#     masked_img[:, y:y + mask_size, x:x + mask_size] = 0
#     return masked_img, (x, y, mask_size)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        resnet18 = models.resnet18(weights=None)  
        
        # Convert generator to list before slicing
        self.encoder = nn.Sequential(*list(resnet18.children())[:-2])
        
    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # Output: 256, 6, 6
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # Output: 128, 12, 12
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # Output: 64, 24, 24
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # Output: 32, 48, 48
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),     # Output: 3, 96, 96
            nn.Sigmoid()  # Output is an image with pixel values in [0, 1]
        )
    
    def forward(self, x):
        return self.decoder(x)


class InpaintingModel(nn.Module):
    def __init__(self, pretrained_encoder=None):
        super(InpaintingModel, self).__init__()
        # Encoder
        if pretrained_encoder is not None:
            self.encoder = pretrained_encoder
            print('using pre-trained encoder')
        else:
            self.encoder = Encoder()
            print('using new encoder')
        self.decoder = Decoder()
        

    def forward(self, x):
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        return reconstructed


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),   # Output: (64, 48, 48)
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # Output: (64, 24, 24)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),# Output: (64, 12, 12)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=4, stride=3, padding=1),  # Output: (64, 1, 1) with stride 3
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, kernel_size=4, stride=3, padding=1)
        )

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)  # Flatten to (64, 1)
        return x  # Output shape: (64, 1)
