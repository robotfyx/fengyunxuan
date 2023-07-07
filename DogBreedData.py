# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 10:51:33 2023

@author: 99358
"""

from PIL import Image
import torch
from torch.utils.data import Dataset
import os
import pandas as pd

label_csv = pd.read_csv('labels.csv')#存放标签的csv文件
labels = label_csv.set_index('id')#以id为行索引
breed_index = pd.read_excel('breed_index.xls')
breed_index_dict = dict(zip(breed_index['breed'],breed_index['index']))

class DogBreedData(Dataset):
    
    def __init__(self, datadir, transform=None):
        self.datadir = datadir#数据集路径
        self.img_names = os.listdir(self.datadir)
        self.transform = transform#图片转tensor方式
    
    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, idx):
        img_name = self.img_names[idx]#图片名称
        path = os.path.join(self.datadir, img_name)
        img = Image.open(path)
        if self.transform:
            m_img = self.transform(img)
        
        label = breed_index_dict[labels['breed'][img_name[:-4]]]
        m_label = torch.tensor(label)
        
        return m_img,m_label
        