import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from glob import glob
from utils import Degradation
import cv2
import numpy as np

class CelebADataset(Dataset):
    def __init__(self, train=True,get_path=False,path=None):
        self.transform = transforms.Compose([
            transforms.Resize((512,512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.get_path =get_path
        if train:
            self.image_paths = glob(f'{path}/*/set_C/*.jpg')
        else:
            self.image_paths = glob(f'{path}/*/*/*.jpg')
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        if self.get_path:
            return img_path,image
        return image
    
class VGGFaceDataset(Dataset):
    def __init__(self, train=True,get_path=False,path=None):
        self.transform = transforms.Compose([
            transforms.Resize((512,512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.get_path =get_path
        if train:
            self.image_paths = glob(f'{path}/*/set_C/*.png')
        else:
            self.image_paths = glob(f'{path}/*/*/*.png')
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        if self.get_path:
            return img_path,image
        return image
    

class FFHQDataset(Dataset):
    def __init__(self, train=True,path=None):
        self.transform = transforms.Compose([
            transforms.Resize((512,512)),
            # transforms.RandomResizedCrop(size=512,scale=(0.1,1.0),ratio=(1.0,1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        self.image_paths = glob(f'{path}/*.png')
   
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image




