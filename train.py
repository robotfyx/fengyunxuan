# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 13:02:31 2023

@author: 99358
"""

import torch
from torch import nn
from DogBreedData import DogBreedData
from torch.utils.data import DataLoader,random_split
from torchvision import transforms,models
from train_utils import try_gpu,get_dataloader_workers,savelist
import time
from sklearn.metrics import accuracy_score

trans = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.CenterCrop((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def get_net(device):#定义网络，使用预训练好的resnet34网络，最后自己定义新输出网络为120类
    finetune_net = nn.Sequential()
    finetune_net.features = models.resnet34(pretrained=True)
    # 定义一个新的输出网络，共有120个输出类别
    finetune_net.output_new = nn.Sequential(nn.Linear(1000, 256),
                                            nn.ReLU(),
                                            nn.Linear(256, 120))
    # 将模型参数分配给用于计算的CPU或GPU
    finetune_net = finetune_net.to(device)
    # 冻结参数
    for param in finetune_net.features.parameters():
        param.requires_grad = False
    return finetune_net

def test_model(model, test_data_loader, loss, device):
    model.eval()
    with torch.no_grad():
        l_sum = 0.0
        acc = 0
        for j,(imgs,labels) in enumerate(test_data_loader):
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            output = model(imgs)
            
            l = loss(output.float(), labels.long()).sum()
            l_sum += l.item()*labels.size(0)
            acc += accuracy_score(labels.to('cpu'), output.argmax(axis=1).to('cpu'))*labels.size(0)
    return l_sum/len(test_data),acc/len(test_data)

def train_model(epochs, model, train_data_loader, test_data_loader, loss, optimizer, scheduler, device):
    Train_loss = list()
    Test_acc = list()
    Test_loss = list()
    for epoch in range(epochs):
        start = time.time()
        print('epoch:', epoch)
        print('training on', device)
        
        train_loss = 0.0
        
        model.train()
        
        for i,(imgs,labels) in enumerate(train_data_loader):
            imgs = imgs.to(device)
            labels = labels.to(device)
                
            optimizer.zero_grad()
                
            y_pre = model(imgs)
            l = loss(y_pre.float(), labels.long()).sum()
            
            train_loss += l.item()*labels.size(0)
            
            l.backward()
            optimizer.step()
        scheduler.step()
        
        end = time.time()
        
        
        test_loss,test_acc = test_model(
            model = model, 
            test_data_loader = test_data_loader, 
            loss = loss, 
            device = device
            )
        
        print('train_loss:', train_loss/len(train_data))
        print('test_loss:', test_loss)
        print('test_acc:', test_acc)
        print('spend time:',end-start,'s')
        print('*'*40)
        
        Train_loss.append(train_loss/len(train_data))
        Test_acc.append(test_loss)
        Test_loss.append(test_acc)
        
    torch.save(model, 'model.pth')
    savelist(Train_loss, 'result/Train_loss1.txt')
    savelist(Test_acc, 'result/Test_acc1.txt')
    savelist(Test_loss, 'result/Test_loss1.txt')    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == '__main__':
    whole_data = DogBreedData('train', transform=trans)
    train_size = int(0.8*len(whole_data))#训练集图片数
    test_size = len(whole_data)-train_size#测试集图片数
    train_data,test_data = random_split(whole_data, [train_size,test_size])
    
    train_data_loader = DataLoader(
        train_data,
        batch_size = 128,
        shuffle=True,
        num_workers=get_dataloader_workers()
        )#训练集
    test_data_loader = DataLoader(
        test_data,
        batch_size = 128,
        shuffle=True,
        num_workers=get_dataloader_workers()
        )#测试集
    
    device = try_gpu()#使用GPU
    
    net = get_net(device)#加载网络
    model = nn.DataParallel(net).to(device)
    
    loss = nn.CrossEntropyLoss()#交叉熵损失函数
    optimizer = torch.optim.SGD((param for param in model.parameters() if param.requires_grad),
                                lr=1e-3,
                                momentum=0.9)#优化模型
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    epochs = 15#训练次数
    
    train_model(
        epochs = epochs, 
        model = model, 
        train_data_loader = train_data_loader, 
        test_data_loader = test_data_loader, 
        loss = loss, 
        optimizer = optimizer, 
        scheduler = scheduler, 
        device = device
        )
    
    