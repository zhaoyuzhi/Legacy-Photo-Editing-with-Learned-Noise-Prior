# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 15:04:36 2019

@author: ZHAO Yuzhi
"""

import cv2
import numpy as np
from skimage import io
from skimage import measure
from skimage import transform
from skimage import color

# Compute the mean-squared error between two images
def MSE(srcpath, dstpath, scale = 256):
    scr = cv2.imread(srcpath)
    dst = cv2.imread(dstpath)
    scr = transform.resize(scr, (scale, scale))
    dst = transform.resize(dst, (scale, scale))
    mse = measure.compare_mse(scr, dst)
    return mse

# Compute the normalized root mean-squared error (NRMSE) between two images
def NRMSE(srcpath, dstpath, mse_type = 'Euclidean', scale = 256):
    scr = cv2.imread(srcpath)
    #scr = cv2.imread(srcpath, flags = cv2.IMREAD_GRAYSCALE)        # only for grayscale images
    #scr = scr[:, :, np.newaxis]                                    # only for grayscale images
    #scr = np.concatenate((scr, scr, scr), axis = 2)                # only for grayscale images
    dst = cv2.imread(dstpath)
    scr = transform.resize(scr, (scale, scale))
    dst = transform.resize(dst, (scale, scale))
    nrmse = measure.compare_nrmse(scr, dst, norm_type = mse_type)
    return nrmse

# Compute the peak signal to noise ratio (PSNR) for an image
def PSNR(srcpath, dstpath, scale = 256):
    scr = cv2.imread(srcpath)
    #scr = cv2.imread(srcpath, flags = cv2.IMREAD_GRAYSCALE)        # only for grayscale images
    #scr = scr[:, :, np.newaxis]                                    # only for grayscale images
    #scr = np.concatenate((scr, scr, scr), axis = 2)                # only for grayscale images
    dst = cv2.imread(dstpath)
    scr = transform.resize(scr, (scale, scale))
    dst = transform.resize(dst, (scale, scale))
    psnr = measure.compare_psnr(scr, dst)
    return psnr

# Compute the mean structural similarity index between two images
def SSIM(srcpath, dstpath, RGBinput = True, scale = 256):
    scr = cv2.imread(srcpath)
    #scr = cv2.imread(srcpath, flags = cv2.IMREAD_GRAYSCALE)        # only for grayscale images
    #scr = scr[:, :, np.newaxis]                                    # only for grayscale images
    #scr = np.concatenate((scr, scr, scr), axis = 2)                # only for grayscale images
    dst = cv2.imread(dstpath)
    scr = transform.resize(scr, (scale, scale))
    dst = transform.resize(dst, (scale, scale))
    ssim = measure.compare_ssim(scr, dst, multichannel = RGBinput)
    return ssim

# If the gray scale needs:
#   gray = color.rgb2gray(scr)
#   dst = color.gray2rgb(gray)
# add these two lines in the functions
'''
path = 'D:\\dataset\\MIT Place 2016\\val_256\\Places365_val_00000106.jpg'
path_zhang = 'C:\\Users\\ZHAO Yuzhi\\Desktop\\dataset\\colorization_results\\2016Zhang\\MITPlace_val_256\\Places365_val_00000106.jpg'
path_pix2pix = 'C:\\Users\\ZHAO Yuzhi\\Desktop\\dataset\\colorization_results\\2017Pix2Pix\\MITPlace_val_256\\Places365_val_00000106.jpg'

mse1 = NRMSE(path, path_zhang)
mse2 = NRMSE(path, path_pix2pix)
psnr1 = PSNR(path, path_zhang)
psnr2 = PSNR(path, path_pix2pix)
ssim1 = SSIM(path, path_zhang)
ssim2 = SSIM(path, path_pix2pix)
print(mse1)
print(mse2)
print(psnr1)
print(psnr2)
print(ssim1)
print(ssim2)
'''
