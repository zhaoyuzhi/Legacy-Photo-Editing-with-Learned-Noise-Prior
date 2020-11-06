# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 16:04:34 2018

@author: ZHAO Yuzhi
"""

import argparse
import os
import time
import datetime
import numpy as np
import torch
import torchvision as tv
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

import network
import dataset
from PIL import Image

# This code is not run
def main():
    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type = int, default = 50, help = 'number of epochs of training')
    parser.add_argument('--batch_size', type = int, default = 16, help = 'size of the batches')
    parser.add_argument('--lr', type = float, default = 0.0005, help='SGD: learning rate')
    parser.add_argument('--momentum', type=float, default = 0.9, help='SGD: momentum')
    parser.add_argument('--weight_decay', type = float, default = 5e-5, help = 'SGD: weight-decay')
    parser.add_argument('--weight_decline', type = int, default = 10, help = 'weight decline for optimizer')
    parser.add_argument('--num_workers', type = int, default = 8, help = 'number of cpu threads to use during batch generation')
    parser.add_argument('--img_height', type = int, default = 224, help = 'size of image height')
    parser.add_argument('--img_width', type = int, default = 224, help = 'size of image width')
    parser.add_argument('--pre_train', type = bool, default = False, help = 'pre-train ot not')
    parser.add_argument('--checkpoint_interval', type = int, default = 5, help = 'interval between model checkpoints')
    opt = parser.parse_args()
    print(opt)
    cudnn.benchmark = True
    # ----------------------------------------
    #       Network training parameters
    # ----------------------------------------
    # Loss functions
    criterion = torch.nn.CrossEntropyLoss().cuda()

    # Initialize generator
    if opt.pre_train == True:
        net = network.VGG16(init_weights = True).cuda()
    else:
        net = torch.load('VGG16_epoch20_loss0.9908_batchsize16.pkl')

    # Optimizers
    optimizer = torch.optim.SGD(net.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)

    def adjust_learning_rate(optimizer, epoch, opt):
        """Sets the learning rate to the initial LR decayed by 10 every 100 epochs"""
        lr = opt.lr * (0.5 ** (epoch // opt.weight_decline))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    # ----------------------------------------
    #             Network dataset
    # ----------------------------------------

    # Define the image list
    imglist = dataset.text_readlines("ILSVRC2012_train_224.txt")
    stringlist = dataset.text_readlines("mapping_string.txt")
    scalarlist = dataset.text_readlines("mapping_scalar.txt")

    # Define the dataset
    ImageNet_train_dataset = dataset.ImageNet_train(imglist, stringlist, scalarlist)

    # Define the dataloader
    dataloader = DataLoader(ImageNet_train_dataset, batch_size = opt.batch_size, shuffle = True, num_workers = opt.num_workers, pin_memory = True)
    print('The overall numbers of images:', len(ImageNet_train_dataset))

    prev_time = time.time()
    
    # ----------------------------------------
    #            Training and Testing
    # ----------------------------------------
    for epoch in range(opt.epochs):
        for batch_idx, (data, target) in enumerate(dataloader):
            data = data.cuda()
            target = target.cuda()
            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + batch_idx
            batches_left = opt.epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds = batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            print("\r[Epoch %d/%d] [Batch %d/%d] [loss: %.5f] time_left: %s" %
                            (epoch, opt.epochs, batch_idx, len(dataloader),
                            loss.item(), time_left))
        
        if (epoch + 1) % opt.checkpoint_interval == 0:
            torch.save(net, 'VGG16_epoch%d_loss%.4f_batchsize%d.pkl' % ((epoch + 20 + 1), loss.item(), opt.batch_size))
            print('The trained model is successfully saved!')
        
        adjust_learning_rate(optimizer, epoch, opt)

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()
