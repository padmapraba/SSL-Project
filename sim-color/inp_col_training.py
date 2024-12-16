import cv2
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import STL10
from torchvision.models import resnet18
import torch.nn as nn
from colorization import Colorization, RGB2LabTransform, STL10ColorizationDataset, EarlyStopping
import torch
from torchvision import datasets, transforms

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Data augmentation and normalization for training
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load STL10 dataset
stl10_pretrain = STL10ColorizationDataset(root='../data', split='train+unlabeled', download=True, transform=transform)

# DataLoader to feed batches for training
pretrain_loader = DataLoader(stl10_pretrain, batch_size=64, shuffle=True)

for L_channel, AB_channels in pretrain_loader:
    print(f"L_channel shape: {L_channel.shape}, AB_channels shape: {AB_channels.shape}")
    break  # Just checking one batch

# Load pre-trained SimSiam backbone - this was trained separately using SimSiam self-supervised learning
backbone = resnet18()
backbone = nn.Sequential(*list(backbone.children())[:-2])  # Remove final layers to use as feature extractor
backbone.load_state_dict(torch.load('backbone_weights_6.pth'))  # Load weights from SimSiam pre-training
backbone = backbone.to(device)

# Initialize colorization model using the pre-trained SimSiam backbone as the encoder
colorization_model = Colorization(backbone)  # The backbone weights are frozen and used as feature extractor
colorization_model = colorization_model.to(device)

# Training loop for colorization task
num_epochs = 100
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(colorization_model.parameters(), lr=1e-3)

early_stop = EarlyStopping(paitence=15, min_delta=0.000001)
# Create models directory if it doesn't exist
import os
import matplotlib.pyplot as plt
import json
os.makedirs('models', exist_ok=True)

# Lists to store loss values for plotting
train_losses = []

for epoch in range(num_epochs):
    total_loss = 0.0  # Initialize total loss for this epoch
    num_batches = 0   # Keep track of the number of batches

    # Training loop for the current epoch
    for L_channel, AB_channels in pretrain_loader:
        # Move data to the same device as the model
        L_channel = L_channel.to(device)
        AB_channels = AB_channels.to(device)
        
        L_channel_rgb = L_channel.repeat(1, 3, 1, 1)  # Shape: [batch_size, 3, 96, 96]

        # Forward pass through colorization model using SimSiam features
        predicted_AB = colorization_model(L_channel_rgb)

        # Compute loss
        loss = criterion(predicted_AB, AB_channels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate loss and count batches
        total_loss += loss.item()
        num_batches += 1

    # Calculate average loss for the epoch
    average_loss = total_loss / num_batches
    train_losses.append(average_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss}')

    # Save losses after each epoch
    with open('training_losses.json', 'w') as f:
        json.dump(train_losses, f)

    early_stop(average_loss)
    if early_stop.early_stop:
        print("Early Stopping Triggered. No improves in Loss for the last 10 epochs")
        break
    
    if (epoch + 1) % 10 == 0:
        torch.save(colorization_model.state_dict(), f'models/simsiam_colorization_model_weights_epoch_{epoch+1}.pth')

    if average_loss < early_stop.best_loss:
        best_model_weights = colorization_model.state_dict()

# Plot training loss using saved data
with open('training_losses.json', 'r') as f:
    saved_losses = json.load(f)

plt.figure(figsize=(10, 6))
plt.plot(saved_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.legend()
plt.grid(True)
plt.savefig('training_loss_plot.png')
plt.close()

torch.save(colorization_model.state_dict(), 'models/simsiam_colorization_model_weights_final.pth')
torch.save(best_model_weights, 'models/simsiam_colorization_best_model_weights_final.pth')

import numpy as np
from skimage import color
import matplotlib.pyplot as plt

def visualize_colorization(L_channel, predicted_AB, ground_truth_AB):
    batch_size = L_channel.shape[0]

    for i in range(batch_size):
        # Convert model's output (predicted_AB) to RGB for each sample in the batch
        colorized_image = lab_to_rgb(L_channel[i], predicted_AB[i])

        # Convert the ground truth to RGB for each sample
        ground_truth_rgb = lab_to_rgb(L_channel[i], ground_truth_AB[i])

        # Display the colorized image and the ground truth (visualization code)
        plt.subplot(1, 2, 1)
        plt.imshow(colorized_image)
        plt.title('Predicted Colorization')

        plt.subplot(1, 2, 2)
        plt.imshow(ground_truth_rgb)
        plt.title('Ground Truth')

        plt.show()
        print(f"Predicted AB min: {predicted_AB[i].min():.2f}, max: {predicted_AB[i].max():.2f}")
        print(f"Ground Truth AB min: {ground_truth_AB[i].min():.2f}, max: {ground_truth_AB[i].max():.2f}")

# Convert LAB to RGB
def lab_to_rgb(L_channel, AB_channels):
    # Ensure L_channel has shape [96, 96] and scale it appropriately
    L_channel = L_channel.squeeze().cpu().numpy() * 255
    L_channel = L_channel 

    # Ensure AB_channels has shape [2, 96, 96] and transpose it to [96, 96, 2]
    AB_channels = AB_channels.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
    AB_channels = (AB_channels * 255) - 128

    # Concatenate L and AB channels to form LAB image
    lab_image = np.concatenate((L_channel[:, :, np.newaxis], AB_channels), axis=-1)

    # Convert LAB to RGB using a library like skimage
    rgb_image = color.lab2rgb(lab_image)
    
    return rgb_image

# Test visualization
visualize_colorization(L_channel, predicted_AB, AB_channels)
