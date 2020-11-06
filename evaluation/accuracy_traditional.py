# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 19:36:00 2018

@author: yzzhao2
"""

import numpy as np
import os
from PIL import Image
from indexes_traditional import *

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

def get_jpgs(path):
    # read a folder, return the image name
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(filespath)
    return ret

# save a list to a txt
def text_save(content, filename, mode = 'a'):
    # Try to save a list variable in txt file.
    file = open(filename, mode)
    for i in range(len(content)):
        file.write(str(content[i]) + '\n')
    file.close()

# Traditional indexes accuracy for dataset
def Dset_Acuuracy(imglist, refpath, basepath):
    # Define the list saving the accuracy
    nrmselist = []
    psnrlist = []
    ssimlist = []
    nrmseratio = 0
    psnrratio = 0
    ssimratio = 0

    # Compute the accuracy
    for i in range(len(imglist)):
        # Full imgpath
        imgname = imglist[i]
        print(imgname)
        refimgpath = os.path.join(refpath, imgname)
        #imgpath = os.path.join(basepath, imgname)
        imgpath = os.path.join(basepath, imgname[:-5] + '.JPEG')
        # Compute the traditional indexes
        nrmse = NRMSE(refimgpath, imgpath)
        psnr = PSNR(refimgpath, imgpath)
        ssim = SSIM(refimgpath, imgpath)
        nrmselist.append(nrmse)
        psnrlist.append(psnr)
        ssimlist.append(ssim)
        nrmseratio = nrmseratio + nrmse
        psnrratio = psnrratio + psnr
        ssimratio = ssimratio + ssim
        print('The %dth image: nrmse: %f, psnr: %f, ssim: %f' % (i, nrmse, psnr, ssim))
    nrmseratio = nrmseratio / len(imglist)
    psnrratio = psnrratio / len(imglist)
    ssimratio = ssimratio / len(imglist)

    return nrmselist, psnrlist, ssimlist, nrmseratio, psnrratio, ssimratio
    
if __name__ == "__main__":
    
    # Read all names
    imglist = get_jpgs('F:\\submitted papers\\my papers\\SCGAN v1\\Colorization results (image original resolution)\\ground truth')

    # Define reference path
    refpath = 'C:\\Users\\yzzha\\Desktop\\dataset\\ILSVRC2012_val_256'

    # Define imgpath
    basepath = 'F:\\submitted papers\\my papers\\SCGAN v1\\Colorization results (image)\\ablation study\\GAN8_BN_batchsize8_epoch20_new\\ILSVRC2012_val_256'
    
    nrmselist, psnrlist, ssimlist, nrmseratio, psnrratio, ssimratio = Dset_Acuuracy(imglist, refpath, basepath)

    print('The overall results: nrmse: %f, psnr: %f, ssim: %f' % (nrmseratio, psnrratio, ssimratio))
