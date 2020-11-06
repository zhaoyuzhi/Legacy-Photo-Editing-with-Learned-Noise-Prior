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

class DomainTransferDataset(Dataset):
    def __init__(self, opt):
        super(DomainTransferDataset, self).__init__()
        self.opt = opt
        self.imglist_A = utils.get_files(opt.baseroot_A)
        self.imglist_B = utils.get_files(opt.baseroot_B)
        self.len_A = len(self.imglist_A)
        self.len_B = len(self.imglist_B)
    
    def imgcrop(self, img):
        H, W = img.shape
        # scaled size should be greater than opts.crop_size
        if H < W:
            if H < self.opt.crop_size:
                H_out = self.opt.crop_size
                W_out = int(math.floor(W * float(H_out) / float(H)))
                img = cv2.resize(img, (W_out, H_out))
        else: # W_out < H_out
            if W < self.opt.crop_size:
                W_out = self.opt.crop_size
                H_out = int(math.floor(H * float(W_out) / float(W)))
                img = cv2.resize(img, (W_out, H_out))
        # randomly crop
        rand_h = random.randint(0, max(0, H - self.opt.imgsize))
        rand_w = random.randint(0, max(0, W - self.opt.imgsize))
        img = img[rand_h:rand_h + self.opt.imgsize, rand_w:rand_w + self.opt.imgsize, :]
        return img

    def __getitem__(self, index):

        ## Image A
        random_A = random.randint(0, self.len_A - 1)
        imgpath_A = self.imglist_A[random_A]
        img_A = cv2.imread(imgpath_A, cv2.IMREAD_GRAY)
        # image cropping
        img_A = self.imgcrop(img_A)
        
        ## Image B
        random_B = random.randint(0, self.len_B - 1)
        imgpath_B = self.imglist_B[random_B]
        img_B = cv2.imread(imgpath_B, cv2.IMREAD_GRAY)
        # image cropping
        img_B = self.imgcrop(img_B)

        # To tensor (grayscale)
        img_A = torch.from_numpy(img_A.astype(np.float32) / 255.0).unsqueeze(0).contiguous()
        img_B = torch.from_numpy(img_B.astype(np.float32) / 255.0).unsqueeze(0).contiguous()

        return img_A, img_B
    
    def __len__(self):
        return min(self.len_A, self.len_B)

class InpaintDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.imglist = utils.get_jpgs(opt.baseroot)
        self.masklist = utils.get_jpgs(opt.maskroot)

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, index):

        # image read
        imgname = self.imglist[index]                                       # name of one image
        imgpath = os.path.join(self.opt.baseroot, imgname)                  # path of one image
        img = cv2.imread(imgpath, flags = cv2.IMREAD_GRAYSCALE)             # read one image (grayscale)
        # image resize
        img = cv2.resize(img, (self.opt.crop_size, self.opt.crop_size))
        gt = img.copy()

        # mask
        maskname = str(random.randint(1, len(self.masklist))) + '.png'
        maskpath = os.path.join(self.opt.maskroot, maskname)
        mask = cv2.imread(maskpath, flags = cv2.IMREAD_GRAYSCALE)           # read one image (grayscale)
        # randomly crop
        rand_h = random.randint(0, max(0, mask.shape[0] - self.opt.crop_size))
        rand_w = random.randint(0, max(0, mask.shape[1] - self.opt.crop_size))
        mask = mask[rand_h:rand_h + self.opt.crop_size, rand_w:rand_w + self.opt.crop_size]

        # design input image
        masked_img = (img.astype(np.float64) / 255) * (1 - mask.astype(np.float64) / 255)
        if self.opt.noise_aug:   ### For short exposure pixels, it is 4 times than long exposure pixels
            noise = np.random.normal(loc = 0.0, scale = self.opt.noise_level, size = masked_img.shape)
            masked_img = masked_img + noise
            masked_img = np.clip(masked_img, 0, 1)
        
        # normalize output image
        gt = gt.astype(np.float64) / 255

        # normalization
        masked_img = torch.from_numpy(masked_img.astype(np.float32)).unsqueeze(0).contiguous()
        gt = torch.from_numpy(gt.astype(np.float32)).unsqueeze(0).contiguous()
        return masked_img, gt

class InpaintDataset_val(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.imglist = utils.get_jpgs(opt.baseroot)
        self.masklist = utils.get_jpgs(opt.maskroot)

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, index):

        # image read
        imgname = self.imglist[index]                                       # name of one image
        imgpath = os.path.join(self.opt.baseroot, imgname)                  # path of one image
        img = cv2.imread(imgpath, flags = cv2.IMREAD_GRAYSCALE)             # read one image (grayscale)
        # image resize
        img = cv2.resize(img, (self.opt.crop_size, self.opt.crop_size))
        gt = img.copy()

        # mask
        maskname = str(random.randint(1, len(self.masklist))) + '.png'
        maskpath = os.path.join(self.opt.maskroot, maskname)
        mask = cv2.imread(maskpath, flags = cv2.IMREAD_GRAYSCALE)           # read one image (grayscale)
        # randomly crop
        rand_h = random.randint(0, max(0, mask.shape[0] - self.opt.crop_size))
        rand_w = random.randint(0, max(0, mask.shape[1] - self.opt.crop_size))
        mask = mask[rand_h:rand_h + self.opt.crop_size, rand_w:rand_w + self.opt.crop_size]

        # design input image
        masked_img = (img.astype(np.float64) / 255) * (1 - mask.astype(np.float64) / 255)
        if self.opt.noise_aug:   ### For short exposure pixels, it is 4 times than long exposure pixels
            noise = np.random.normal(loc = 0.0, scale = self.opt.noise_level, size = masked_img.shape)
            masked_img = masked_img + noise
            masked_img = np.clip(masked_img, 0, 1)
        
        # normalize output image
        gt = gt.astype(np.float64) / 255

        # normalization
        masked_img = torch.from_numpy(masked_img.astype(np.float32)).unsqueeze(0).contiguous()
        gt = torch.from_numpy(gt.astype(np.float32)).unsqueeze(0).contiguous()
        return masked_img, gt, imgname

class Generate_mask_and_masked_image_Dataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.imglist = utils.get_jpgs(opt.baseroot)
        self.masklist = utils.get_jpgs(opt.maskroot)

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, index):

        # image read
        imgname = self.imglist[index]                                       # name of one image
        imgpath = os.path.join(self.opt.baseroot, imgname)                  # path of one image
        img = cv2.imread(imgpath, flags = cv2.IMREAD_GRAYSCALE)             # read one image (grayscale)
        # image resize
        img = cv2.resize(img, (self.opt.crop_size, self.opt.crop_size))
        gt = img.copy()

        # mask
        maskname = str(random.randint(1, len(self.masklist))) + '.png'
        maskpath = os.path.join(self.opt.maskroot, maskname)
        mask = cv2.imread(maskpath, flags = cv2.IMREAD_GRAYSCALE)           # read one image (grayscale)
        # randomly crop
        rand_h = random.randint(0, max(0, mask.shape[0] - self.opt.crop_size))
        rand_w = random.randint(0, max(0, mask.shape[1] - self.opt.crop_size))
        mask = mask[rand_h:rand_h + self.opt.crop_size, rand_w:rand_w + self.opt.crop_size]

        # design input image
        masked_img = (img.astype(np.float64) / 255) * (1 - mask.astype(np.float64) / 255)
        if self.opt.noise_aug:   ### For short exposure pixels, it is 4 times than long exposure pixels
            noise = np.random.normal(loc = 0.0, scale = self.opt.noise_level, size = masked_img.shape)
            masked_img = masked_img + noise
            masked_img = np.clip(masked_img, 0, 1)
        
        # normalize output image
        gt = gt.astype(np.float64) / 255
        mask = mask.astype(np.float64) / 255

        # normalization
        masked_img = torch.from_numpy(masked_img.astype(np.float32)).unsqueeze(0).contiguous()
        gt = torch.from_numpy(gt.astype(np.float32)).unsqueeze(0).contiguous()
        mask = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).contiguous()
        return masked_img, gt, mask, imgname

class Generate_mask_and_masked_image_Legacy_Image_Dataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.imglist = utils.get_jpgs(opt.baseroot)
        self.masklist = utils.get_jpgs(opt.maskroot)

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, index):

        # image read
        imgname = self.imglist[index]                                       # name of one image
        imgpath = os.path.join(self.opt.baseroot, imgname)                  # path of one image
        img = cv2.imread(imgpath, flags = cv2.IMREAD_GRAYSCALE)             # read one image (grayscale)
        # image resize
        img = cv2.resize(img, (self.opt.crop_size, self.opt.crop_size))
        gt = img.copy()

        # mask
        maskname = str(random.randint(1, len(self.masklist))) + '.png'
        maskpath = os.path.join(self.opt.maskroot, maskname)
        mask = cv2.imread(maskpath, flags = cv2.IMREAD_GRAYSCALE)           # read one image (grayscale)
        # randomly crop
        mask = cv2.resize(mask, (self.opt.crop_size, self.opt.crop_size))
        rand_h = random.randint(0, max(0, mask.shape[0] - self.opt.mask_size))
        rand_w = random.randint(0, max(0, mask.shape[1] - self.opt.mask_size))
        mask = mask[rand_h:rand_h + self.opt.mask_size, rand_w:rand_w + self.opt.mask_size]

        computational_mask = np.zeros(img.shape, dtype = np.float64)
        rand_h = random.randint(0, max(0, computational_mask.shape[0] - self.opt.mask_size))
        rand_w = random.randint(0, max(0, computational_mask.shape[1] - self.opt.mask_size))
        computational_mask[rand_h:rand_h + self.opt.mask_size, rand_w:rand_w + self.opt.mask_size] = mask

        # design input image
        masked_img = (img.astype(np.float64) / 255) * (1 - computational_mask.astype(np.float64) / 255)
        if self.opt.noise_aug:   ### For short exposure pixels, it is 4 times than long exposure pixels
            noise = np.random.normal(loc = 0.0, scale = self.opt.noise_level, size = masked_img.shape)
            masked_img = masked_img + noise
            masked_img = np.clip(masked_img, 0, 1)
        
        # normalize output image
        gt = gt.astype(np.float64) / 255
        computational_mask = computational_mask.astype(np.float64) / 255

        # normalization
        masked_img = torch.from_numpy(masked_img.astype(np.float32)).unsqueeze(0).contiguous()
        gt = torch.from_numpy(gt.astype(np.float32)).unsqueeze(0).contiguous()
        computational_mask = torch.from_numpy(computational_mask.astype(np.float32)).unsqueeze(0).contiguous()
        return masked_img, gt, computational_mask, imgname

if __name__ == "__main__":

    imgpath = './example.JPEG'                                          # path of one image
    img = cv2.imread(imgpath, flags = cv2.IMREAD_GRAYSCALE)             # read one image (grayscale)
    # image resize
    img = cv2.resize(img, (256, 256))
    gt = img.copy()

    # mask
    maskpath = './huahen/processed/3.png'
    mask = cv2.imread(maskpath, flags = cv2.IMREAD_GRAYSCALE)           # read one image (grayscale)
    # randomly crop
    rand_h = random.randint(0, max(0, mask.shape[0] - 256))
    rand_w = random.randint(0, max(0, mask.shape[1] - 256))
    mask = mask[rand_h:rand_h + 256, rand_w:rand_w + 256]

    # design input image
    masked_img = (img.astype(np.float64) / 255) * (1 - mask.astype(np.float64) / 255)
    if 1:   ### For short exposure pixels, it is 4 times than long exposure pixels
        noise = np.random.normal(loc = 0.0, scale = 0.03, size = masked_img.shape)
        masked_img = masked_img + noise
        masked_img = np.clip(masked_img, 0, 1)
    
    # normalize output image
    masked_img = (masked_img * 255).astype(np.uint8)

    show = np.concatenate((gt, masked_img), axis = 1)
    cv2.imshow('show', show)
    cv2.waitKey(0)
