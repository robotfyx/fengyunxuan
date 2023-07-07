# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 18:50:34 2023

@author: 99358
"""

import torch
from torchvision import transforms
from PIL import Image
from matplotlib import pyplot as plt

def try_gpu(i=0):
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')
device = try_gpu()

trans = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.CenterCrop((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

net = torch.load('model.pth')

img = Image.open('baofu.png')
x = trans(img).reshape((1,3,224,224))
x.to(device)
y = net(x)
print(y.argmax(axis=1).item())
plt.imshow(img)
