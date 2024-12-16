import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
from torchvision.models import resnet18
import random
import numpy as np
import os

# Set device and random seeds
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device:', device)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Define the Shared Encoder (ResNet-18 Backbone)
class SharedEncoder(nn.Module):
    def __init__(self, rotation_model_weights=None):
        super(SharedEncoder, self).__init__()
        self.backbone = resnet18(weights=None)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])  # Remove the final classification layer

        # Load rotation weights if provided
        if rotation_model_weights:
            state_dict = torch.load(rotation_model_weights)
            self.backbone.load_state_dict(state_dict, strict=False)
            print("Loaded rotation model weights into the shared encoder.")

    def forward(self, x):
        return self.backbone(x)

# Classification Model for downstream task
class ClassificationModel(nn.Module):
    def __init__(self, encoder, num_classes=101):  # CALTECH101 has 101 classes
        super(ClassificationModel, self).__init__()
        self.encoder = encoder
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match ResNet input size
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
])

# Load CALTECH101 dataset
def load_caltech101():
    full_dataset = datasets.Caltech101(root='./data', download=True, transform=transform)
    
    # Split into train and test sets (80% train, 20% test)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    # Split train into train and validation (80% train, 20% validation)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    return train_dataset, val_dataset, test_dataset

def main():
    # Load datasets
    train_dataset, val_dataset, test_dataset = load_caltech101()
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Initialize model with pretrained backbone
    rotation_weights_path = 'model_weights/rotation_pretext_cross_entropy_model.pth'  # Update with your path
    backbone = SharedEncoder(rotation_model_weights=rotation_weights_path)
    classification_model = ClassificationModel(encoder=backbone.backbone).to(device)

    if torch.cuda.device_count() > 1:
        classification_model = nn.DataParallel(classification_model)

    # Training parameters
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classification_model.parameters(), lr=0.001)
    epochs = 150
    early_stopping_patience = 15
    best_val_loss = float('inf')
    no_improve_epochs = 0

    # Training loop
    for epoch in range(epochs):
        classification_model.train()
        running_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = classification_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}")

        # Validation
        classification_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_inputs, val_targets in val_loader:
                val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                val_outputs = classification_model(val_inputs)
                val_loss += criterion(val_outputs, val_targets).item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Validation Loss: {avg_val_loss:.4f}")

        # Model checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(classification_model.state_dict(), 'caltech101_classification_model.pth')
            print(f"Model saved at Epoch {epoch+1}")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        # Early stopping
        if no_improve_epochs >= early_stopping_patience:
            print("Early stopping triggered")
            break

    # Final evaluation
    classification_model.eval()
    correct = 0
    top_3_correct = 0
    top_5_correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = classification_model(images)
            _, predicted = outputs.topk(5, dim=1)

            total += labels.size(0)
            correct += (predicted[:, 0] == labels).sum().item()
            top_3_correct += (predicted[:, :3] == labels.unsqueeze(1)).any(dim=1).sum().item()
            top_5_correct += (predicted == labels.unsqueeze(1)).any(dim=1).sum().item()

    print(f'Top-1 Accuracy: {100 * correct / total:.2f}%')
    print(f'Top-3 Accuracy: {100 * top_3_correct / total:.2f}%')
    print(f'Top-5 Accuracy: {100 * top_5_correct / total:.2f}%')

if __name__ == "__main__":
    main()
