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



import torch.nn as nn
import torch.nn.functional as F

'''
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x




class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        # 첫번째층
        # ImgIn shape=(?, 28, 28, 1)
        #    Conv     -> (?, 28, 28, 32)
        #    Pool     -> (?, 14, 14, 32)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        # 두번째층
        # ImgIn shape=(?, 14, 14, 32)
        #    Conv      ->(?, 14, 14, 64)
        #    Pool      ->(?, 7, 7, 64)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        # 전결합층 7x7x64 inputs -> 10 outputs
        self.fc = torch.nn.Linear(7 * 7 * 64, 10, bias=True)

        # 전결합층 한정으로 가중치 초기화
        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)   # 전결합층을 위해서 Flatten
        out = self.fc(out)
        return out





'''

# Simple CNN
class CNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=100):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, # 입력 채널 수. 흑백 이미지일 경우 1, RGB 값을 가진 이미지일 경우 3 
            out_channels=8, #  출력 채널 수
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
        self.fc1 = nn.Linear(1024, num_classes)  # 64*16
        # RuntimeError: Function AddmmBackward returned an invalid gradient at index 1 - got [64, 784] but expected shape compatible with [64, 1024]



    def forward(self, x):
        #print("연산 전", x.size())  # ([64, 3, 32, 32])
        x = F.relu(self.conv1(x))
        #print("conv1(x) 연산 후", x.size())  # ([64, 8, 32, 32]) 
        x = self.pool(x)
        #print("pool(x) 연산 후",x.size())  # ([64, 8, 16, 16])
        x = F.relu(self.conv2(x))
        #print("conv2(x) 연산 후",x.size())  # ([64, 16, 16, 16])
        x = self.pool(x)
        #print("pool(x) 연산 후",x.size()) # ([64, 16, 8, 8])
        x = x.reshape(x.shape[0], -1)
        #print("reshape 연산 후",x.size()) # ([64, 16, 8, 8])
        x = self.fc1(x)
        #print("fc1(x) 연산 후",x.size()) # ([64, 100])
        return x
        