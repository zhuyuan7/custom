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



# Simple CNN
class CNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=100):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, # 입력 채널 수. 흑백 이미지일 경우 1, RGB 값을 가진 이미지일 경우 3 
            out_channels=8, #  출력 채널 수 !! 사용자 정함
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
        self.fc1 = nn.Linear(1024, num_classes)  # 64*16 (16*8*8) ---> nn.Linear(1024, num_classes) 
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
        



        