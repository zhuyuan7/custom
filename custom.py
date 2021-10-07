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


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transforms=None): # 내가 필요한 것들 (데이터 셋을 가져와서 선처리)
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transforms = transforms
        

    
    def __len__(self): # 데이터 셋의 길이 반환
        return len(self.img_labels)

    
    def __getitem__(self, idx):  # 데이터 셋에서  한 개의 데이터를 가져오는 함수 정의 , #샘플 반환(이미지와 라벨 dict형태로반환)
        #img_file = glob.glob(os.path.join(img_dir, "*.jpg"))
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx,0])
        
        img = io.imread(img_path)
        y_label = torch.tensor(int(self.img_labels.iloc[idx,2]))
        if self.transforms is not None:  
            image = self.transforms(image)
          
        return (image, y_label)

        
# Simple CNN
class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=8,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.fc1 = nn.Linear(16 * 7 * 7, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x


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
dataset = CustomImageDataset(
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

model = CNN(in_channels=in_channels, num_classes=num_classes).to(device)
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
