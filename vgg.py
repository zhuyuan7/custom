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



opt = parse_opt()

# CIFAR 100   32*32*3
class VGG16(nn.Module):
    def __init__(self):
          

        super(VGG16, self).__init__()

        # CONV layer 1  CR CR P
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size = 3, stride = 1, padding =1)
        self.relu1_1 = nn.ReLU(inplace=False)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding =1)
        self.relu1_2 = nn.ReLU(inplace=False)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)) 


        # CONV layer 2 CR CR P
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding =1)
        self.relu2_1 = nn.ReLU(inplace=False)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size = 3, stride = 1, padding =1)
        self.relu2_2 = nn.ReLU(inplace=False)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))


        # CONV layer 3 CR CR CR P
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding =1)
        self.relu3_1 = nn.ReLU(inplace=False)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding =1)
        self.relu3_2 = nn.ReLU(inplace=False)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding =1)
        self.relu3_3 = nn.ReLU(inplace=False)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))


         # CONV layer 4 CR CR CR P
        self.conv4_1 = nn.Conv2d(256,  512, kernel_size = 3, stride = 1, padding =1)
        self.relu4_1 = nn.ReLU(inplace=False)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding =1)
        self.relu4_2 = nn.ReLU(inplace=False)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding =1)
        self.relu4_3 = nn.ReLU(inplace=False)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        

        # CONV layer 5 CR CR CR P
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding =1)
        self.relu5_1 = nn.ReLU(inplace=False)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding =1)
        self.relu5_2 = nn.ReLU(inplace=False)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding =1)
        self.relu5_3 = nn.ReLU(inplace=False)
        self.pool5 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))


        self.fc1 = nn.Linear(512, 256)
        self.relu6_1 = nn.ReLU(inplace = True)
        self.dropout6_1 = nn.Dropout(p=opt.drop_prob)
        self.fc2 = nn.Linear(512, 256)
        self.relu6_2 = nn.ReLU(inplace = True)
        self.dropout6_2 = nn.Dropout(p=opt.drop_prob)
        self.fc3 = nn.Linear(opt.batch_size, opt.num_classes)



    def forward(self, x):   
        print("연산 전", x.size())  # 연산 전 torch.Size([64, 3, 32, 32])
        x = self.relu1_1(self.conv1_1(x))  
        x = self.relu1_2(self.conv1_2(x))  
        x = self.pool1(x)  # output: 16 * 16 * 64
        print("conv1", x.size())  # conv1 torch.Size([64, 64, 16, 16])


        x = self.relu2_1(self.conv2_1(x))
        x = self.relu2_2(self.conv2_2(x))
        x = self.pool2(x)
        print("conv2", x.size())  # conv2 torch.Size([64, 128, 8, 8])


        x = self.relu3_1(self.conv3_1(x))
        x = self.relu3_2(self.conv3_2(x))
        x = self.relu3_3(self.conv3_3(x))
        x = self.pool3(x)
        print("conv3", x.size())  # conv3 torch.Size([64, 256, 4, 4])


        x = self.relu4_1(self.conv4_1(x))
        print("conv4_1", x.size())  # conv4_1 torch.Size([64, 521, 4, 4])
        x = self.relu4_2(self.conv4_2(x))
        print("conv4_2", x.size())  # conv4_2 torch.Size([64, 512, 4, 4])
        x = self.relu4_3(self.conv4_3(x))
        print("conv4_3", x.size()) # conv4_3 torch.Size([64, 512, 4, 4])
        x = self.pool4(x)
        print("conv4", x.size())  # conv4 torch.Size([64, 512, 2, 2])

        x = self.relu5_1(self.conv5_1(x))
        x = self.relu5_2(self.conv5_2(x))
        x = self.relu5_3(self.conv5_3(x))
        x = self.pool5(x)
        print("conv5", x.size()) # conv5 torch.Size([64, 512, 1, 1])


        x = x.reshape(x.shape[0], -1)
        print("reshape", x.size())  # reshape torch.Size([64, 512])
        x = self.relu6_1(self.fc1(x))
        print("relu6_1", x.size())  # relu6_1 torch.Size([64, 256])
        x = self.dropout6_1(x)
        print("dropout6_1", x.size())  # dropout6_1 torch.Size([64, 256])
       
       
       # RuntimeError: CUDA error: CUBLAS_STATUS_INVALID_VALUE when calling `cublasSgemm( handle, opa, opb, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc)`
        x = self.relu6_2(self.fc2(x))
        print("relu6_2", x.size()) 
       
        x = self.dropout6_2(x)
        print("dropout6_2", x.size())
        x = self.fc3(x)
        print("fc3", x.size())
       
        return x