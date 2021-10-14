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
from aps import parse_opt


'''
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

### layer 10개 
opt = parse_opt() 

class CNN(nn.Module):
    def __init__(self):
          
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3,8,3,1,1) #32*32*8
        self.pool = nn.MaxPool2d(2,2)

        self.conv2 = nn.Conv2d(8,16,3,1,1) #(x, FH, FW, self.stride, self.pad)
        

        self.conv3 = nn.Conv2d(16,32,3, 1,1)
        

        self.conv4 = nn.Conv2d(32,64, 3, 1,1)
        
        #self.fc1 = nn.Linear(1024, 1024)
        #self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, opt.num_classes)  #64*2*2, num_classes
        # Function AddmmBackward returned an invalid gradient at index 1 - got [64, 64] but expected shape compatible with [64, 256]
        #self.softmax = nn.LogSoftmax

    def forward(self, x):
        #print("연산 전", x.size()) # torch.Size([64, 3, 32, 32])
        x = F.relu(self.conv1(x))
        #print("conv1(x) 연산 후",x.size())   #torch.Size([64, 8, 32, 32])
        x = self.pool(x)
        #print("pool(x) 연산 후",x.size())  # torch.Size([64, 8, 16, 16])
        #x = self.pool1(F.relu(self.conv1(x)))  # 64, 8 , 16, 16

        x = F.relu(self.conv2(x))
        #print("conv2(x) 연산 후",x.size())  #torch.Size([64, 16, 16, 16])
        x = self.pool(x)
        #print("pool() 연산 후",x.size())  # torch.Size([64, 16, 8, 8])
        #x = self.pool2(F.relu(self.conv2(x)))  # 64, 16, 8 ,8

        # x = F.relu(self.conv3(x))
        # #print("conv3(x) 연산 후",x.size())  #torch.Size([64, 32, 8, 8])
        # x = self.pool(x)
        # #print("pool(x) 연산 후",x.size())
        # #x = self.pool3(F.relu(self.conv3(x)))   # 64, 32 , 4 ,4

        # x = F.relu(self.conv4(x))
        # #print("conv4(x) 연산 후",x.size())
        # x = self.pool(x)
        # #print("pool(x) 연산 후",x.size())
        # #x = self.pool4(F.relu(self.conv4(x))) # 64, 64, 2 ,2

        x = x.reshape(x.shape[0], -1)
        #x = x.view(-1, 16384)# the size -1 is inferred from other dimensions
        #print("reshape(x) 연산 후",x.size())  # 1, 16384

        x = F.relu(self.fc3(x))

        #x = F.relu(self.fc2(x))



        #x = F.softmax(self.fc3(x))
        #print("fc1(x) 연산 후",x.size()) # 1, 100  __> 수정 

        return x
#net = CNN().to(device)
#print(net)


'''





opt = parse_opt()

# Simple CNN
class CNN(nn.Module):
    def __init__(self, in_channels=opt.in_channels, num_classes=opt.num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, # 입력 채널 수. 흑백 이미지일 경우 1, RGB 값을 가진 이미지일 경우 3 
            out_channels=64, #  출력 채널 수 !! 사용자 정함
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        
        
        self.conv2 = nn.Conv2d(
            in_channels=64, 
            out_channels=128,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )

        self.conv3 = nn.Conv2d(
            in_channels=128, 
            out_channels=256,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )

        self.conv4 = nn.Conv2d(
            in_channels=256, 
            out_channels=512,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )


        self.conv5 = nn.Conv2d(
            in_channels=512, 
            out_channels=1024,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )

        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)) 
        #self.relu() = nn.ReLU(inplace=False)
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
        x = F.relu(self.conv3(x))
        #print("conv2(x) 연산 후",x.size())  # ([64, 16, 16, 16])
        x = self.pool(x)
        
        x = F.relu(self.conv4(x))
        #print("conv2(x) 연산 후",x.size())  # ([64, 16, 16, 16])
        x = self.pool(x)
        
        x = F.relu(self.conv5(x))
        #print("conv2(x) 연산 후",x.size())  # ([64, 16, 16, 16])
        x = self.pool(x)

        x = x.reshape(x.shape[0], -1)
        #print("reshape 연산 후",x.size()) # ([64, 16, 8, 8])
        x = self.fc1(x)
        #print("fc1(x) 연산 후",x.size()) # ([64, 100])
        return x



'''
Checking accuracy on Training Set
Got 26318 / 50000 with accuracy 52.64
Checking accuracy on Test Set
Got 4446 / 10000 with accuracy 44.46
'''