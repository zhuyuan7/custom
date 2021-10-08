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
import os


# RuntimeError : CUDA error: device-side assert triggered
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} System")


# Hyperparameters
in_channels = 3
num_classes = 100
learning_rate = 1e-3
batch_size = 64
num_epochs = 10

transforms = transforms.Compose(  # 쉽게 말해 우리의 데이터를 전처리하는 패키지.
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
     #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])  # 이미지의 경우 픽셀 값 하나는 0 ~ 255 값을 갖는다. 하지만 ToTensor()로 타입 변경시 0 ~ 1 사이의 값으로 바뀜.

print("ok")   


# Load Data
dataset = custom.CustomImageDataset(
    annotations_file="/home/joo/archive/train.csv",
    img_dir="/home/joo/archive/train_images",
    transforms = transforms
    )

test_dataset = custom.CustomImageDataset(
    annotations_file="/home/joo/archive/test.csv",
    img_dir="/home/joo/archive/test_images",
    transforms = transforms
    )



print(len(dataset), len(test_dataset))


#lengths = [int(len(dataset)*0.8), int(len(dataset)*0.2)]

#train_set, test_set = torch.utils.data.random_split(dataset, lengths)


train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset , batch_size=batch_size, shuffle=True, num_workers=0)





for X, y in train_loader:
    print(X)
    print(y)
    break
    
# Model
#model = model.CNN().to(device)
model = model.CNN(in_channels=in_channels, num_classes=num_classes).to(device)
#model = torchvision.models.wide_resnet50_2(pretrained=True)
#model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()  # nn.CrossEntropyLoss()의 경우 기본적으로 LogSoftmax()가 내장. 
                                   # 실제 값과 예측값의 차이 (dissimilarity) 계산한다는 관점에서 cross-entropy 사용하는 것
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

    #test  시작
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