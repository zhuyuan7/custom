import os 
import glob
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, Compose
from torch.utils.data.dataset import random_split
import torch.optim as optim
from skimage import io
import torch.nn.functional as F 
import model
import custom

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} System")


# Hyperparameters
in_channels = 3
num_classes = 2
learning_rate = 1e-3
batch_size = 64
num_epochs = 10

transforms = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

print("ok")   


# Load Data
dataset = custom.CustomImageDataset(
    annotations_file="/home/joo/archive/train.csv",
    img_dir="/home/joo/archive/train_images",
    transforms = transforms
    )




lengths = [int(len(dataset)*0.8), int(len(dataset)*0.2)]

train_set, test_set = torch.utils.data.random_split(dataset, lengths)
train_loader = DataLoader(dataset = train_set, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(dataset = test_set, batch_size=batch_size, shuffle=True, num_workers=0)

for X, y in train_loader:
    print(X)
    print(y)
    break
    
# Model

model = model.CNN(in_channels=in_channels, num_classes=num_classes).to(device)
#model = torchvision.models.wide_resnet50_2(pretrained=True)
#model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Network
for epoch in range(num_epochs):
    losses = []

    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        losses.append(loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

    print(f"Cost at epoch {epoch} is {sum(losses)/len(losses)}")

# Check accuracy on training to see how good our model is
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(
            f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}"
        )

    model.train()


print("Checking accuracy on Training Set")
check_accuracy(train_loader, model)

print("Checking accuracy on Test Set")
check_accuracy(test_loader, model)