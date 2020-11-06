import os
import math
import random
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
import cv2
from PIL import Image

import utils

class ScribbleColorizationDataset(Dataset):
    def __init__(self, opt):
        #assert opt.mask_type in ALLMASKTYPES
        self.opt = opt
        self.imglist = utils.get_jpgs(opt.baseroot)
        self.stringlist = utils.text_readlines(opt.stringlist)
        self.scalarlist = utils.text_readlines(opt.scalarlist)

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, index):

        # image read
        imgname = self.imglist[index]                                       # name of one image
        imgpath = os.path.join(self.opt.baseroot, imgname)                  # path of one image
        img = Image.open(imgpath).convert('RGB')                            # read one image (RGB)
        img = np.array(img)                                                 # read one image
        # image resize
        img = cv2.resize(img, (self.opt.imgsize, self.opt.imgsize), interpolation = cv2.INTER_CUBIC)
        # grayish
        grayscale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # color map
        color_scribble = self.color_scribble(img = img, color_point = self.opt.color_point, color_width = self.opt.color_width)
        #color_scribble = self.blurish(img = color_scribble, color_blur_width = self.opt.color_blur_width)
        color_scribble = torch.from_numpy(color_scribble.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()

        # normalization
        grayscale = torch.from_numpy(grayscale.astype(np.float32) / 255.0).unsqueeze(0).contiguous()
        img = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()

        ### target part
        stringname = imgname[:9]                                            # category by str: like n01440764
        for index, value in enumerate(self.stringlist):
            if stringname == value:
                target = self.scalarlist[index]                             # target: 1~1000
                target = int(target) - 1                                    # target: 0~999
                target = np.array(target)                                   # target: 0~999
                target = torch.from_numpy(target).long()
                break
        target = 0
        # grayscale: 1 * 256 * 256; img: 3 * 256 * 256; color_scribble: 3 * 256 * 256
        return grayscale, img, color_scribble, target

    def color_scribble(self, img, color_point, color_width):
        height = img.shape[0]
        width = img.shape[1]
        channels = img.shape[2]
        scribble = np.zeros((height, width, channels), np.uint8)

        times = np.random.randint(color_point)
        for i in range(times):
            # random selection
            rand_h = np.random.randint(height)
            rand_w = np.random.randint(width)
            # define min and max
            min_h = rand_h - (color_width - 1) // 2
            max_h = rand_h + (color_width - 1) // 2
            min_w = rand_w - (color_width - 1) // 2
            max_w = rand_w + (color_width - 1) // 2
            min_h = max(min_h, 0)
            min_w = max(min_w, 0)
            max_h = min(max_h, height)
            max_w = min(max_w, width)
            # attach color points
            scribble[min_h:max_h, min_w:max_w, :] = img[rand_h, rand_w, :]

        return scribble
    
    def blurish(self, img, color_blur_width):
        img = cv2.GaussianBlur(img, (color_blur_width, color_blur_width), 0)
        return img
    
class ScribbleColorizationValDataset(Dataset):
    def __init__(self, opt):
        #assert opt.mask_type in ALLMASKTYPES
        self.opt = opt
        self.imglist = utils.get_jpgs(opt.baseroot)

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, index):

        # image read
        imgname = self.imglist[index]                                       # name of one image
        imgpath = os.path.join(self.opt.baseroot, imgname)                  # path of one image
        img = Image.open(imgpath).convert('RGB')                            # read one image (RGB)
        img = np.array(img)                                                 # read one image
        # image resize
        img = cv2.resize(img, (self.opt.imgsize, self.opt.imgsize), interpolation = cv2.INTER_CUBIC)
        # grayish
        grayscale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # color map
        color_scribble = self.color_scribble(img = img, color_point = self.opt.color_point, color_width = self.opt.color_width)
        #color_scribble = self.blurish(img = color_scribble, color_blur_width = self.opt.color_blur_width)
        color_scribble = torch.from_numpy(color_scribble.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()

        # normalization
        grayscale = torch.from_numpy(grayscale.astype(np.float32) / 255.0).unsqueeze(0).contiguous()
        img = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()

        # grayscale: 1 * 256 * 256; img: 3 * 256 * 256; color_scribble: 3 * 256 * 256
        return grayscale, img, color_scribble, imgname

    def color_scribble(self, img, color_point, color_width):
        height = img.shape[0]
        width = img.shape[1]
        channels = img.shape[2]
        scribble = np.zeros((height, width, channels), np.uint8)

        times = np.random.randint(color_point)
        for i in range(times):
            # random selection
            rand_h = np.random.randint(height)
            rand_w = np.random.randint(width)
            # define min and max
            min_h = rand_h - (color_width - 1) // 2
            max_h = rand_h + (color_width - 1) // 2
            min_w = rand_w - (color_width - 1) // 2
            max_w = rand_w + (color_width - 1) // 2
            min_h = max(min_h, 0)
            min_w = max(min_w, 0)
            max_h = min(max_h, height)
            max_w = min(max_w, width)
            # attach color points
            scribble[min_h:max_h, min_w:max_w, :] = img[rand_h, rand_w, :]

        return scribble
    
    def blurish(self, img, color_blur_width):
        img = cv2.GaussianBlur(img, (color_blur_width, color_blur_width), 0)
        return img
    
class ScribbleColorizationValDataset_SingleImg(Dataset):
    def __init__(self, opt):
        #assert opt.mask_type in ALLMASKTYPES
        self.opt = opt

    def __len__(self):
        return 1

    def __getitem__(self, index):

        # image read
        imgname = self.opt.imgroot                                          # name of one image
        img = Image.open(imgname).convert('RGB')                            # read one image (RGB)
        img = np.array(img)                                                 # read one image
        # image resize
        img = cv2.resize(img, (self.opt.imgsize, self.opt.imgsize), interpolation = cv2.INTER_CUBIC)
        # grayish
        grayscale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # color map
        color_scribble = self.color_scribble(img = img, color_point = self.opt.color_point, color_width = self.opt.color_width)
        #color_scribble = self.blurish(img = color_scribble, color_blur_width = self.opt.color_blur_width)
        color_scribble = torch.from_numpy(color_scribble.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()

        # normalization
        grayscale = torch.from_numpy(grayscale.astype(np.float32) / 255.0).unsqueeze(0).contiguous()
        img = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()

        # grayscale: 1 * 256 * 256; img: 3 * 256 * 256; color_scribble: 3 * 256 * 256
        return grayscale, img, color_scribble, imgname

    def color_scribble(self, img, color_point, color_width):
        height = img.shape[0]
        width = img.shape[1]
        channels = img.shape[2]
        scribble = np.zeros((height, width, channels), np.uint8)

        times = np.random.randint(color_point)
        for i in range(times):
            # random selection
            rand_h = np.random.randint(height)
            rand_w = np.random.randint(width)
            # define min and max
            min_h = rand_h - (color_width - 1) // 2
            max_h = rand_h + (color_width - 1) // 2
            min_w = rand_w - (color_width - 1) // 2
            max_w = rand_w + (color_width - 1) // 2
            min_h = max(min_h, 0)
            min_w = max(min_w, 0)
            max_h = min(max_h, height)
            max_w = min(max_w, width)
            # attach color points
            scribble[min_h:max_h, min_w:max_w, :] = img[rand_h, rand_w, :]

        return scribble
    
    def blurish(self, img, color_blur_width):
        img = cv2.GaussianBlur(img, (color_blur_width, color_blur_width), 0)
        return img
    
class ScribbleColorizationValDataset_Known_ColorScribble(Dataset):
    def __init__(self, opt):
        #assert opt.mask_type in ALLMASKTYPES
        self.opt = opt
        self.imglist = utils.get_jpgs(opt.baseroot)

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, index):

        # image read
        imgname = self.imglist[index]                                       # name of one image
        imgpath = os.path.join(self.opt.baseroot, imgname)                  # path of one image
        img = Image.open(imgpath).convert('RGB')                            # read one image (RGB)
        img = np.array(img)                                                 # read one image
        # image resize
        img = cv2.resize(img, (self.opt.imgsize, self.opt.imgsize), interpolation = cv2.INTER_CUBIC)
        # grayish
        grayscale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # color map read
        color_scribble_path = os.path.join(self.opt.colorscribbleroot, imgname.split('.')[0] + '.png')
        color_scribble = Image.open(color_scribble_path).convert('RGB')     # read one image (RGB)
        color_scribble = np.array(color_scribble)                           # read one image
        # image resize
        color_scribble = cv2.resize(color_scribble, (self.opt.imgsize, self.opt.imgsize), interpolation = cv2.INTER_CUBIC)
        #color_scribble = self.blurish(img = color_scribble, color_blur_width = self.opt.color_blur_width)
        color_scribble = torch.from_numpy(color_scribble.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()

        # normalization
        grayscale = torch.from_numpy(grayscale.astype(np.float32) / 255.0).unsqueeze(0).contiguous()
        img = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()

        # grayscale: 1 * 256 * 256; img: 3 * 256 * 256; color_scribble: 3 * 256 * 256
        return grayscale, img, color_scribble, imgname
