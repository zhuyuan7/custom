import os 
import pandas as pd
from skimage import io
import os
from aps import parse_opt

import torch
import torch.nn as nn
import torchvision
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, Compose
from torch.utils.data.dataset import random_split
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F 

# import custom
from custom import CustomImageDataset
from model import CNN


# RuntimeError : CUDA error: device-side assert triggered
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 다른 Gpu 사용 어떻게 하는 지 
print(f"Using {device} System")



def main(opt):
    # RuntimeError : CUDA error: device-side assert triggered


    train_transforms = transforms.Compose(   # 쉽게 말해 우리의 데이터를 전처리하는 패키지. # 이미지의 경우 픽셀 값 하나는 0 ~ 255 값을 갖는다. 
        [transforms.RandomCrop(32, padding=4),  #  랜덤한 부분을 [size, size] 크기로 잘라냄. input 이미지가 output 크기보다 작으면 padding추가 가능
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),   # ToTensor()로 타입 변경시 0 ~ 1 사이의 값으로 바뀜
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])   # -1~ 1 사이로 normalize 함
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])  

    test_transforms = transforms.Compose(   # 쉽게 말해 우리의 데이터를 전처리하는 패키지. # 이미지의 경우 픽셀 값 하나는 0 ~ 255 값을 갖는다. 
        [transforms.ToTensor(),   # ToTensor()로 타입 변경시 0 ~ 1 사이의 값으로 바뀜
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])   # -1~ 1 사이로 normalize 함
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])  

    print("ok")   


    # Load Data
    train_dataset = CustomImageDataset(
        annotations_file="/home/joo/archive/train.csv",
        img_dir="/home/joo/archive/train_images",
        train_transforms = train_transforms
        )

    test_dataset = CustomImageDataset(
        annotations_file="/home/joo/archive/test.csv",
        img_dir="/home/joo/archive/test_images",
        test_transforms = test_transforms
        )
        
    print(len(train_dataset), len(test_dataset))


    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    test_loader = DataLoader(test_dataset , batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)


    # for X, y in test_loader:
    #     print(X) # tesor
    #     print(y) # label
    #     break


    # Model
    model = CNN(in_channels=opt.in_channels, num_classes=opt.num_classes).to(device)
    # model = CNN().to(device)


    


    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()  # nn.CrossEntropyLoss()의 경우 기본적으로 LogSoftmax()가 내장. 
                                    # 실제 값과 예측값의 차이 (dissimilarity) 계산한다는 관점에서 cross-entropy 사용하는 것
    
    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate)
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


    model.train()

    # Train Network
    for epoch in range(opt.num_epochs):

        losses = []

        for batch_idx, (data, targets) in enumerate(train_loader):
            # Get data to cuda if possible
            image = data.to(device=device) # 학습용 data 불러옴 (32*32)
            label = targets.to(device=device) 

            #print("target: ", targets.shape)
            #print("data: ", data.shape)
            
            # forward
            scores = model(image) # score 구함 ########## here
           # print("here========== : ", scores.shape)
            #print("here========== : ", label.shape)
            loss = criterion(scores, label) # score(실제값)과 target(예측값)을 이용해 crossentropyloss 구함

            # print(scores.shape)
            # print(label.shape)
            # print(label)
            # print(scores)
            

            losses.append(loss.item())  # loss값 쌓기

            # backward
            optimizer.zero_grad()  # optimaizer 초기화  
            loss.backward() # 오차만큼 다시 backpropagation 시행

            # gradient descent or adam step
            optimizer.step()  # step 다시 정리 
            # exp_lr_scheduler.step()
            
        print(f"Cost at epoch {epoch} is {sum(losses)/len(losses)}")

        
    print("Checking accuracy on Training Set")
    check_accuracy(train_loader, model)

    print("Checking accuracy on Test Set")
    check_accuracy(test_loader, model)


# Check accuracy on training to see how good our model is
def check_accuracy(test_loader, model):
    num_correct = 0
    num_samples = 0

    #test  시작 
    model.eval()
    

    with torch.no_grad():
        for data, targets in test_loader:
            image = data.to(device=device)
            label = targets.to(device=device)

            scores = model(image)
            #print(scores)  # size, 그냥 프린트하면 값나옴
            _, predictions = scores.max(1)  
            
            num_correct += (predictions == label).sum()  
            num_samples += predictions.size(0)   # batch size를 64로 정했으니까 한 루프당 64/// minibatch = 64  

        print(
            f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}"
        )             # 정확도 

   

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)