import os
import random
import math
import cv2
import numpy as np
import torch
import torch.utils.data as data

import utils

class DomainTransferDataset(data.Dataset):
    def __init__(self, opt):
        super(DomainTransferDataset, self).__init__()
        self.opt = opt
        self.imglist_A = utils.get_files(opt.baseroot_A)
        self.imglist_B = utils.get_files(opt.baseroot_B)
        self.len_A = len(self.imglist_A)
        self.len_B = len(self.imglist_B)
    
    def process_img(self, imgpath, legacy_judgement):
        # read image
        img = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
        H, W = img.shape
        # scaled size should be greater than opts.crop_size
        if H < W:
            if H < self.opt.crop_size:
                H_out = self.opt.crop_size
                W_out = int(math.floor(W * float(H_out) / float(H)))
                img = cv2.resize(img, (W_out, H_out))
            else:
                H_out = H
                W_out = W
        else: # W_out < H_out
            if W < self.opt.crop_size:
                W_out = self.opt.crop_size
                H_out = int(math.floor(H * float(W_out) / float(W)))
                img = cv2.resize(img, (W_out, H_out))
            else:
                H_out = H
                W_out = W
        '''
        # randomly crop
        rand_h = random.randint(0 + H // 10, max(0, H - H // 10 - self.opt.crop_size))
        rand_w = random.randint(0 + W // 10, max(0, W - W // 10 - self.opt.crop_size))
        img = img[rand_h:rand_h + self.opt.crop_size, rand_w:rand_w + self.opt.crop_size]
        img = np.expand_dims(img, axis = 2)
        '''
        # randomly crop
        rand_h = random.randint(0, max(0, H_out - self.opt.crop_size))
        rand_w = random.randint(0, max(0, W_out - self.opt.crop_size))
        img = img[rand_h:rand_h + self.opt.crop_size, rand_w:rand_w + self.opt.crop_size]
        return img

    def __getitem__(self, index):

        ## Image A (clean image)
        random_A = random.randint(0, self.len_A - 1)
        imgpath_A = self.imglist_A[random_A]
        img_A = self.process_img(imgpath_A, False)

        ## Image B (legacy image)
        random_B = random.randint(0, self.len_B - 1)
        imgpath_B = self.imglist_B[random_B]
        img_B = self.process_img(imgpath_B, True)

        # Normalization
        img_A = img_A.astype(np.float32) / 255.0
        img_B = img_B.astype(np.float32) / 255.0
        img_A = torch.from_numpy(img_A).unsqueeze(0).contiguous()
        img_B = torch.from_numpy(img_B).unsqueeze(0).contiguous()

        return img_A, img_B
    
    def __len__(self):
        return min(self.len_A, self.len_B)

class PairedImage_DomainTransferDataset(data.Dataset):
    def __init__(self, opt):
        super(PairedImage_DomainTransferDataset, self).__init__()
        self.opt = opt
        self.imglist = utils.get_jpgs(opt.baseroot_A)
    
    def process_img(self, imgpath1, imgpath2):
        # read image
        img1 = cv2.imread(imgpath1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(imgpath2, cv2.IMREAD_GRAYSCALE)
        H, W = img1.shape
        # scaled size should be greater than opts.crop_size
        if H < W:
            if H < self.opt.crop_size:
                H_out = self.opt.crop_size
                W_out = int(math.floor(W * float(H_out) / float(H)))
                img = cv2.resize(img, (W_out, H_out))
            else:
                H_out = H
                W_out = W
        else: # W_out < H_out
            if W < self.opt.crop_size:
                W_out = self.opt.crop_size
                H_out = int(math.floor(H * float(W_out) / float(W)))
                img = cv2.resize(img, (W_out, H_out))
            else:
                H_out = H
                W_out = W
        # randomly crop
        rand_h = random.randint(0, max(0, H_out - self.opt.crop_size))
        rand_w = random.randint(0, max(0, W_out - self.opt.crop_size))
        img1 = img1[rand_h:rand_h + self.opt.crop_size, rand_w:rand_w + self.opt.crop_size]
        img2 = img2[rand_h:rand_h + self.opt.crop_size, rand_w:rand_w + self.opt.crop_size]
        return img1, img2

    def __getitem__(self, index):

        ## Read image
        imgname = self.imglist[index]
        imgpath_A = os.path.join(self.opt.baseroot_A, imgname)
        imgpath_B = os.path.join(self.opt.baseroot_B, imgname)
        img_A, img_B = self.process_img(imgpath_A, imgpath_B)

        # Normalization
        img_A = img_A.astype(np.float32) / 255.0
        img_B = img_B.astype(np.float32) / 255.0
        img_A = torch.from_numpy(img_A).unsqueeze(0).contiguous()
        img_B = torch.from_numpy(img_B).unsqueeze(0).contiguous()

        return img_A, img_B
    
    def __len__(self):
        return len(self.imglist)
