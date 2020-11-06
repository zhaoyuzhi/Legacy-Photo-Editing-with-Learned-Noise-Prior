# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 12:03:52 2018

@author: yzzhao2
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from skimage import color
from torchvision import transforms
import cv2

def text_readlines(filename):
    # Try to read a txt file and return a list.Return [] if there was a mistake.
    try:
        file = open(filename, 'r')
    except IOError:
        error = []
        return error
    content = file.readlines()
    # This for loop deletes the EOF (like \n)
    for i in range(len(content)):
        content[i] = content[i][:len(content[i])-1]
    file.close()
    return content

class ImageNet_train(Dataset):
    def __init__(self, root, stringlist, scalarlist):                       # the inputs should be lists
        self.imgs = root
        self.stringlist = stringlist
        self.scalarlist = scalarlist
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        imgpath = self.imgs[index]                                          # path of one image
        colorimg = Image.open(imgpath)                                      # read one image
        colorimg = colorimg.convert('RGB')
        img = self.transform(colorimg)
        
        stringname = imgpath.split('/')[-1][:9]                             # category by str: like n01440764
        for index, value in enumerate(self.stringlist):
            if stringname == value:
                target = self.scalarlist[index]                             # target: 1~1000
                target = int(target) - 1                                    # target: 0~999
                target = np.array(target, dtype = np.int64)
                target = torch.from_numpy(target)
        return img, target
