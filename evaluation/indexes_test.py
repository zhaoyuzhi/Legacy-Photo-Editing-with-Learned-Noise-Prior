# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 15:04:36 2019

@author: ZHAO Yuzhi
"""

from indexes_traditional import *
from indexes_new import *
import numpy as np
import math

# read a txt expect EOF
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

# Only evaluate one algorithm at one time
if __name__ == '__main__':

    # Read the name of two testing datasets
    imglist_ILSVRC2012_val = text_readlines('./indexes_files/ILSVRC2012_val_name.txt')
    imglist_MITPlace_val = text_readlines('./indexes_files/MITPlace_val_name.txt')
    average_NRMSE_ILSVRC2012_val = 0
    average_NRMSE_MITPlace_val = 0
    average_PSNR_ILSVRC2012_val = 0
    average_PSNR_MITPlace_val = 0
    average_SSIM_ILSVRC2012_val = 0
    average_SSIM_MITPlace_val = 0

    # The ground truth path
    srcbase1 = 'D:\\dataset\\ILSVRC2012\\ILSVRC2012_val_256\\'
    srcbase2 = 'D:\\dataset\\MIT Place 2016\\val_256\\'

    # Choose an algorithm to test
    algolist = ['Larsson', 'Iizuka', 'Zhang', 'Isola', 'Me']
    algo = 'Zhang'
    # If algo == 'Me', then you should define the variable 'choice_of_me'
    choice_of_me = 'GAN6_8_40'
    # Define the base path 
    if algo == 'Larsson':
        dstbase = 'C:\\Users\\ZHAO Yuzhi\\Desktop\\dataset\\colorization_results\\Larsson\\'
    if algo == 'Iizuka':
        dstbase = 'C:\\Users\\ZHAO Yuzhi\\Desktop\\dataset\\colorization_results\\Iizuka\\'
    if algo == 'Zhang':
        dstbase = 'C:\\Users\\ZHAO Yuzhi\\Desktop\\dataset\\colorization_results\\Zhang\\'
    if algo == 'Isola':
        dstbase = 'C:\\Users\\ZHAO Yuzhi\\Desktop\\dataset\\colorization_results\\Isola\\'
    if algo == 'Me':
        if choice_of_me == 'GAN6_8_40':
            dstbase = 'C:\\Users\\ZHAO Yuzhi\\Desktop\\dataset\\colorization_results\\GAN6_BN_batchsize8_epoch40_new\\'
        if choice_of_me == 'GAN6_8_20':
            dstbase = 'C:\\Users\\ZHAO Yuzhi\\Desktop\\dataset\\colorization_results\\GAN6_BN_batchsize8_epoch40_new\\'

    # Compute the traditional indexes
    for i in range(len(imglist_ILSVRC2012_val)):
        srcpath = srcbase1 + imglist_ILSVRC2012_val[i]
        dstpath = dstbase + 'ILSVRC2012_val_256\\' + imglist_ILSVRC2012_val[i]
        average_NRMSE_ILSVRC2012_val += NRMSE(srcpath, dstpath, mse_type = 'Euclidean')
        average_PSNR_ILSVRC2012_val += PSNR(srcpath, dstpath)
        average_SSIM_ILSVRC2012_val += SSIM(srcpath, dstpath, RGBinput = True)
    average_NRMSE_ILSVRC2012_val = average_NRMSE_ILSVRC2012_val / len(imglist_ILSVRC2012_val)
    average_PSNR_ILSVRC2012_val = average_PSNR_ILSVRC2012_val / len(imglist_ILSVRC2012_val)
    average_SSIM_ILSVRC2012_val = average_SSIM_ILSVRC2012_val / len(imglist_ILSVRC2012_val)

    for i in range(len(imglist_MITPlace_val)):
        srcpath = srcbase2 + imglist_MITPlace_val[i]
        dstpath = dstbase + 'MITPlace_val_256\\' + imglist_MITPlace_val[i]
        average_NRMSE_MITPlace_val += NRMSE(srcpath, dstpath, mse_type = 'Euclidean')
        average_PSNR_MITPlace_val += PSNR(srcpath, dstpath)
        average_SSIM_MITPlace_val += SSIM(srcpath, dstpath, RGBinput = True)
    average_NRMSE_MITPlace_val = average_NRMSE_MITPlace_val / len(imglist_MITPlace_val)
    average_PSNR_MITPlace_val = average_PSNR_MITPlace_val / len(imglist_MITPlace_val)
    average_SSIM_MITPlace_val = average_SSIM_MITPlace_val / len(imglist_MITPlace_val)
