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
        image = io.imread(img_path)

        #image = Image.open(self.image_ids[index]).convert('RGB')
        y_label = torch.tensor(int(self.img_labels.iloc[idx,2]))
        if self.transforms is not None:
            image = self.transforms(image)
          
        return (image, y_label)

        
