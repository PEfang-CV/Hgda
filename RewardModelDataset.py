# !/usr/bin/env python3
"""
Function: Reward Model Dataset
Author: TyFang
Date: 2023/11/29

"""
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import random

class RewardModelDataset(Dataset):

    def __init__(self, txt_file,root):
        self.root=root
        self.frame= open(txt_file).readlines()
        self.transform = transforms.Compose([
        transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
        
        self.transform2 = transforms.Compose([
        transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    def __getitem__(self, idx):
        imageName=self.frame[idx][:-1]

        first=self.root+'/1/'
        second=self.root+'/2/'
        third=self.root+'/3/'
        forth=self.root+'/4/'
        labelMask=self.root+'/label/'
        # print(imageName)
        firstImage = Image.open(first+imageName).convert('RGB')
        secondImage=Image.open(second+imageName).convert('RGB')
        thirdImage=Image.open(third+imageName).convert('RGB')
        forthImage = Image.open(forth+imageName).convert('RGB')
        labelImage=Image.open(labelMask+imageName).convert('RGB')

        firstImage = self.transform(firstImage)
        secondImage = self.transform(secondImage)
        thirdImage = self.transform(thirdImage)
        forthImage = self.transform(forthImage)
        labelImage = self.transform2(labelImage)
        
        sample = {'first': firstImage,'second':secondImage,'third':thirdImage,'forth':forthImage,'label':labelImage}

        return sample

    def __len__(self):
        return len(self.frame)