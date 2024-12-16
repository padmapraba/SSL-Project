import torch.nn as nn
import torch


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