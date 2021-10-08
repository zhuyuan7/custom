import os 
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
from model import CNN
# import custom
from custom import CustomImageDataset
import os


# RuntimeError : CUDA error: device-side assert triggered
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 다른 Gpu 사용 어떻게 하는 지 
print(f"Using {device} System")


# Hyperparameters
in_channels = 3
num_classes = 100
learning_rate = 1e-3  #10 의 -3승
batch_size = 64  # 2의 승수 사용  한 minibache당 데이터수가 64, 남는 거는 자유롭게 
num_epochs = 10

transforms = transforms.Compose(   # 쉽게 말해 우리의 데이터를 전처리하는 패키지. # 이미지의 경우 픽셀 값 하나는 0 ~ 255 값을 갖는다. 
    [transforms.ToTensor(),   # ToTensor()로 타입 변경시 0 ~ 1 사이의 값으로 바뀜
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])   # -1~ 1 사이로 normalize 함
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])  

print("ok")   


# Load Data
dataset = CustomImageDataset(
    annotations_file="/home/joo/archive/train.csv",
    img_dir="/home/joo/archive/train_images",
    transforms = transforms
    )

test_dataset = CustomImageDataset(
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
    print(X) # tesor
    print(y) # label
    break
    
# Model
model = CNN(in_channels=in_channels, num_classes=num_classes).to(device)


# Loss and optimizer
criterion = nn.CrossEntropyLoss()  # nn.CrossEntropyLoss()의 경우 기본적으로 LogSoftmax()가 내장. 
                                   # 실제 값과 예측값의 차이 (dissimilarity) 계산한다는 관점에서 cross-entropy 사용하는 것
optimizer = optim.Adam(model.parameters(), lr=learning_rate)



model.train()
# Train Network
for epoch in range(num_epochs):
    losses = []

    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device=device) # 학습용 data 불러옴 (32*32)
        targets = targets.to(device=device) 

        # forward
        scores = model(data) # score 구함
        loss = criterion(scores, targets) # score(실제값)과 target(예측값)을 이용해 crossentropyloss 구함
        losses.append(loss.item())  # loss값 쌓기

        # backward
        optimizer.zero_grad()  # optimaizer 초기화  
        loss.backward() # 오차만큼 다시 backpropagation 시행

        # gradient descent or adam step
        optimizer.step()  # step 다시 정리 

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
            print(scores)  # size, 그냥 프린트하면 값나옴
            _, predictions = scores.max(1)  # scores. max(1) 1이 뭔지 알아야하고, 왜 아웃풋 값 1이 뭘 의미하는 지!!!
            
            num_correct += (predictions == y).sum() # 
            num_samples += predictions.size(0)   # batch size를 64로 정했으니까 한 루프당 64/// minibatch = 64  

        print(
            f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}"
        )             # 정확도 

    #model.train()


print("Checking accuracy on Training Set")
check_accuracy(train_loader, model)

print("Checking accuracy on Test Set")
check_accuracy(test_loader, model)