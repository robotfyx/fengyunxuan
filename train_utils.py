# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 14:11:52 2023

@author: 99358
"""

import torch
import sys
from torch import nn


def try_gpu(i=0):
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def get_dataloader_workers():
    """使用4个进程来读取数据"""
    return 0 if sys.platform.startswith('win') else 4

def savelist(L, name):
    a = open(name, 'w')
    for i in L:
        i = str(i)
        a.write(i)
        a.write('\n')
    a.close()