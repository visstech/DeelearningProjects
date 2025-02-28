import warnings
warnings.filterwarnings('ignore') 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch.optim as optim
import torch.nn.functional as F
import streamlit as st

# Set dataset paths
train_dir = "/content/PlantDataset/train"
valid_dir = "/content/PlantDataset/valid"
test_dir = "/content/PlantDataset/test"

Diseases_classes = os.listdir(train_dir)
print('Disease classes:', Diseases_classes)
print("Total number of classes:", len(Diseases_classes))

# Check for GPU
def get_default_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_device(data, device):
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device
        
    def __iter__(self):
        for b in self.dataloader:
            yield to_device(b, self.device)
        
    def __len__(self):
        return len(self.dataloader)

device = get_default_device()
print('Using device:', device)

# Define CNN Model
class CNN_NeuralNet(nn.Module):
    def __init__(self, in_channels, num_diseases):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2))

        with torch.no_grad():
            sample_input = torch.zeros(1, in_channels, 64, 64)
            sample_output = self.conv1(sample_input)
            sample_output = self.conv2(sample_output)
            sample_output = self.conv3(sample_output)
            flattened_size = sample_output.view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, num_diseases)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.classifier(out)
        return out

# Initialize and train model
model = CNN_NeuralNet(3, len(Diseases_classes)).to(device)
print('Model:', model)

# Data transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Load datasets
train_dataset = ImageFolder(train_dir, transform=transform)
val_dataset = ImageFolder(valid_dir, transform=transform)

train_loader = DeviceDataLoader(DataLoader(train_dataset, batch_size=8, shuffle=True, pin_memory=True), device)
val_loader = DeviceDataLoader(DataLoader(val_dataset, batch_size=8, pin_memory=True), device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#GradScaler initialization
scaler = torch.amp.GradScaler("cuda" if torch.cuda.is_available() else "cpu")

# Training function
def train_model(model, train_loader, val_loader, epochs=5):
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # Specify device_type in autocast
            
            with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Val Loss: {val_loss/len(val_loader):.4f}, Val Accuracy: {100 * correct/total:.2f}%")

# Train and save model
train_model(model, train_loader, val_loader, epochs=5)
torch.save(model.state_dict(), "plant_disease_cnn.pth")
print('Model saved successfully.')


# Streamlit UI
def streamlit_app():
    st.title("🌿 Plant Disease Detection App 🌿")
    st.write("Upload an image of a plant leaf to detect disease.")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        image = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)
            prediction = Diseases_classes[predicted.item()]
        st.write(f"### 🌱 Prediction: {prediction}")

if __name__ == '__main__':
    streamlit_app()
