# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 19:36:00 2018

@author: yzzhao2
"""

import numpy as np
import os
from PIL import Image
import torch
import torch.nn as nn
import torchvision as tv
from torchvision import transforms
import network

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

# VGG-16 accuracy for one image
# input: one imgpath, a VGG-16 model and the ground truth label
# output: top1 and top5 accuracy (0 or 1)
def Classification_Acuuracy(imgpath, model, label):
    # Read the images
    transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
            ])
    colorimg = Image.open(imgpath)
    colorimg = colorimg.convert('RGB')
    target = transform(colorimg)
    target = target.unsqueeze_(dim = 0)
    target = target.cuda()

    # Evaluate the accuracy
    out = model(target)
    out = torch.softmax(out, 1)
    maxk = max((1, 5))
    _, pred = out.topk(maxk, 1, True, True)
    pred_top5 = pred.cpu().numpy().squeeze()
    
    # Compare the result and label; -1 because targetlist is from 1~1000
    label = int(label) - 1
    if pred_top5[0] == label:
        top1 = 1
    else:
        top1 = 0
    top5 = 0
    for i in range(len(pred_top5)):
        if pred_top5[i] == label:
            top5 = 1
    
    return top1, top5

# VGG-16 accuracy for dataset
def Dset_Acuuracy(vgg16, imglist, targetlist, basepath):
    # Define the list saving the accuracy
    top1list = []
    top5list = []
    top1ratio = 0
    top5ratio = 0

    # Compute the accuracy
    for i in range(len(imglist)):
        # Full imgpath
        imgname = imglist[i]
        imgpath = os.path.join(basepath, imgname)
        # Seek for the index; -1 because image name is from 1~50000
        index = imgpath.split('\\')[-1][-10:-5]
        index = int(index) - 1
        # Compute the top-1 and top-5 accuracy
        top1, top5 = Classification_Acuuracy(imgpath, vgg16, targetlist[index])
        top1list.append(top1)
        top5list.append(top5)
        top1ratio = top1ratio + top1
        top5ratio = top5ratio + top5
        print('The %dth image: top1: %d, top5: %d' % (i, top1, top5))
    top1ratio = top1ratio / len(imglist)
    top5ratio = top5ratio / len(imglist)
    
    return top1list, top5list, top1ratio, top5ratio
    
if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Pre-trained VGG-16
    vgg16 = torch.load('VGG16_epoch55_loss0.0226_batchsize16.pkl').cuda()
    vgg16.eval()
    # Read all names
    #imglist = text_readlines('./other_files/ILSVRC2012_10k_name.txt')
    #imglist = get_jpgs('C:\\Users\\yzzha\\Desktop\\dataset\\ILSVRC2012_val_256')
    imglist = get_jpgs('F:\\submitted papers\\my papers\\SCGAN v2\\Colorization results (image original resolution)\\ChromaGAN')

    # Define target list of validation set 50000 images (index: 0~49999)
    targetlist = text_readlines('./label_files/imagenet_2012_validation_scalar.txt')
    
    # Define imgpath
    #basepath = 'E:\\code\\Legacy Photo Editing\\data\\ILSVRC2012_val_256_DnCNN_CIC_colorized(final_step)'
    basepath = 'F:\\submitted papers\\my papers\\SCGAN v2\\Colorization results (image original resolution)\\ChromaGAN'
    
    top1list, top5list, top1ratio, top5ratio = Dset_Acuuracy(vgg16, imglist, targetlist, basepath)
    
    print('The overall results: top1ratio: %f, top5ratio: %f' % (top1ratio, top5ratio))
    