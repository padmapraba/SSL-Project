import cv2
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import STL10
from torchvision.models import resnet18
import torch.nn as nn
from inpainting import Encoder, Decoder, InpaintingModel, Discriminator, mask_image

# Data augmentation and normalization for training
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load STL10 dataset
train_dataset = datasets.STL10(root='../data', split='train+unlabeled', transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)

if torch.cuda.is_available():
    device = torch.device("cuda")  # Use NVIDIA GPU
    print('cuda')
elif torch.backends.mps.is_available():
    device = torch.device("mps")   # Use Apple's Metal (for M1/M2 Macs)
    print('mps')
else:
    device = torch.device("cpu") 
    print('cpu')

# Load the SimSiam backbone
PATH = 'backbone_weights_6.pth'
backbone = resnet18(weights=None)
backbone = nn.Sequential(*list(backbone.children())[:-1])
backbone.load_state_dict(torch.load(PATH))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
epochs = 100
mask_size = 32  # Size of the mask for inpainting
batch_size = 64
learning_rate = 1e-3

# L2 loss
criterion_l2 = nn.MSELoss()
criterion_bce = nn.BCELoss()

# Move models to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize models
generator = InpaintingModel(pretrained_encoder=backbone).to(device)
discriminator = Discriminator().to(device)

optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002)

for epoch in range(epochs):
    for batch_idx, (real_images, _) in enumerate(train_loader):
        real_images = real_images.to(device)
        
        # Create masked images and mask info using your masking function
        mask_size = 32
        masked_images, mask_info_list = zip(*[mask_image(img, mask_size) for img in real_images])
        masked_images = torch.stack(masked_images).to(device)
        
        # ----------- Discriminator Training -----------
        optimizer_D.zero_grad()
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # Discriminator on real images
        real_output = torch.sigmoid(discriminator(real_images))
        d_loss_real = criterion_bce(real_output, real_labels)

        # Discriminator on fake (reconstructed) images
        reconstructed_images = generator(masked_images)
        
        # Ensure reconstructed images are properly sized before passing to discriminator
        if reconstructed_images.size() != real_images.size():
            # Option 1: Interpolate to match size
            reconstructed_images = nn.functional.interpolate(reconstructed_images, size=real_images.size()[2:])
            
            # Option 2: Center crop if reconstructed is larger
            # reconstructed_images = transforms.CenterCrop(real_images.size()[2:])(reconstructed_images)
            
            # Option 3: Pad if reconstructed is smaller
            # reconstructed_images = nn.functional.pad(
            #     reconstructed_images,
            #     (0, real_images.size(3) - reconstructed_images.size(3),
            #      0, real_images.size(2) - reconstructed_images.size(2))
            # )
        # Detach reconstructed images for discriminator training
        fake_output = torch.sigmoid(discriminator(reconstructed_images.detach()))
        d_loss_fake = criterion_bce(fake_output, fake_labels)

        # Total discriminator loss
        loss_discriminator = (d_loss_real + d_loss_fake) / 2
        loss_discriminator.backward()
        optimizer_D.step()

        # ----------- Generator Training -----------
        optimizer_G.zero_grad()

        # Adversarial loss for the generator
        fake_output = torch.sigmoid(discriminator(reconstructed_images))
        adversarial_loss_gen = criterion_bce(fake_output, real_labels)

        # Focus on reconstruction loss (L2) in the masked region only
        loss_l2 = 0  # Initialize L2 loss
        for i in range(batch_size):
            x, y, size = mask_info_list[i]  # Get mask info for each image
            
            # Real masked region from original image
            real_masked_region = real_images[i, :, y:y + size, x:x + size]
            
            # Reconstructed region from generated image
            reconstructed_masked_region = reconstructed_images[i, :, y:y + size, x:x + size]
            
            # L2 loss between real and reconstructed masked regions
            loss_l2 += criterion_l2(reconstructed_masked_region, real_masked_region)

        loss_l2 /= batch_size  # Average L2 loss across the batch
        
        # Total generator loss (L2 + adversarial)
        total_loss = loss_l2 + 0.001 * adversarial_loss_gen
        total_loss.backward()
        optimizer_G.step()

        # ----------- Print Statistics -----------
        if (batch_idx + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(train_loader)}], '
                  f'L2 Loss: {loss_l2.item():.4f}, Adversarial Loss (Gen): {adversarial_loss_gen.item():.4f}, '
                  f'Total Loss: {total_loss.item():.4f}, Discriminator Loss: {loss_discriminator.item():.4f}')
    
    # Save model checkpoints at intervals
    if (epoch + 1) % 10 == 0:
        torch.save(generator.state_dict(), f'models/simsiam_inpainting_model_gen_weights_epoch_{epoch+1}.pth')
        torch.save(discriminator.state_dict(), f'models/simsiam_inpainting_model_dis_weights_epoch_{epoch+1}.pth')

torch.save({
    'model_state_dict': generator.state_dict(),
    'optimizer_state_dict': optimizer_G.state_dict(),
}, 'models/simsiam_inpainting_model_gen_weights_final.pth')

torch.save({
    'model_state_dict': discriminator.state_dict(),
    'optimizer_state_dict': optimizer_D.state_dict(),
}, 'models/simsiam_inpainting_model_dis_weights_final.pth')

print("Training Complete.")

transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load the STL10 test dataset
test_dataset = datasets.STL10(root='../data', split='test', transform=transform, download=True)

# Define the test data loader
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

# Path to the saved generator model weights
from collections import OrderedDict
PATH = 'models/simsiam_inpainting_model_gen_weights_epoch_100.pth'

generator = InpaintingModel().to(device)
# Load the state dictionary from the checkpoint
checkpoint = torch.load(PATH)

# Adjust the state dict keys
new_state_dict = OrderedDict()
for k, v in checkpoint.items():
    new_key = k.replace("encoder.", "encoder.encoder.")  # Modify based on your structure
    new_state_dict[new_key] = v

# Load the modified state dictionary
generator.load_state_dict(new_state_dict, strict=False)

backbone = generator.encoder

# Set model to evaluation mode
generator.eval()

print("Generator model loaded successfully.")

import matplotlib.pyplot as plt
import torch

# Assuming `generator` is your trained generator/inpainting model and `test_loader` is the STL10 test dataset loader
generator.eval()

# Example of showing the ground truth, masked, and predicted inpainted images for 10 images
dataiter = iter(test_loader)  # Use test_loader to visualize unseen data
num_images_to_inpaint = 20
mask_size = 32  # Adjust as per your need

for i in range(num_images_to_inpaint):
    # Get next batch of images and apply mask
    images, _ = next(dataiter)
    images = images.to(device)  # Send images to device (GPU/CPU)
    
    # Apply mask to the first image of the batch
    masked_image, mask_info = mask_image(images[0], mask_size)  # Returns masked image and mask info
    
    plt.figure(figsize=(9, 3))

    # Original ground truth image
    plt.subplot(1, 3, 1)
    plt.imshow(images[0].permute(1, 2, 0).cpu().numpy())  # Convert tensor to (H, W, C) for display
    plt.title("GroundTruth")
    plt.axis('off')
    
    # Masked image
    plt.subplot(1, 3, 2)
    plt.imshow(masked_image.permute(1, 2, 0).cpu().numpy())
    plt.title("Masked")
    plt.axis('off')
    
    # Reconstruct masked image using the generator model
    with torch.no_grad():
        reconstructed = generator(masked_image.unsqueeze(0))  # Add batch dimension for generator input
        reconstructed = reconstructed.squeeze()  # Remove batch dimension after generator processing
    
    # Replace the masked region in the original image with the inpainted (reconstructed) region
    x, y, size = mask_info  # Coordinates and size of the mask
    inpainted_image = images[0].clone()  # Clone the original image to preserve unmasked regions
    inpainted_image[:, y:y + size, x:x + size] = reconstructed[:, y:y + size, x:x + size]  # Replace masked region
    
    # Show the inpainted (reconstructed) image
    plt.subplot(1, 3, 3)
    plt.imshow(inpainted_image.permute(1, 2, 0).cpu().numpy())  # Convert tensor to (H, W, C)
    plt.title("Inpainted")
    plt.axis('off')

    plt.tight_layout()
    plt.show()
