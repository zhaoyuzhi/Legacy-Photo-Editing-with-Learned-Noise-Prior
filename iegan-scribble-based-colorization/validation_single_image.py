import os
import argparse
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

import utils
import network
import dataset

if __name__ == "__main__":

    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    # Dataset parameters
	# General parameters
    parser.add_argument('--finetune_path', type = str, \
        default = "./models/scribble_colorization_GAN_epoch30_batchsize16.pth", \
            help = 'the load name of models')
    parser.add_argument('--batch_size', type = int, default = 1, help = 'size of the batches')
    parser.add_argument('--save_path', type = str, default = './results', help = 'images save path')
    # Network parameters
    parser.add_argument('--in_channels', type = int, default = 1, help = 'input RGB image')
    parser.add_argument('--scribble_channels', type = int, default = 3, help = 'input scribble image')
    parser.add_argument('--out_channels', type = int, default = 3, help = 'output RGB image')
    parser.add_argument('--start_channels', type = int, default = 64, help = 'latent channels')
    parser.add_argument('--pad', type = str, default = 'reflect', help = 'the padding type')
    parser.add_argument('--activ_g', type = str, default = 'lrelu', help = 'the activation type')
    parser.add_argument('--activ_d', type = str, default = 'lrelu', help = 'the activation type')
    parser.add_argument('--norm_g', type = str, default = 'none', help = 'normalization type')
    parser.add_argument('--norm_d', type = str, default = 'bn', help = 'normalization type')
    parser.add_argument('--init_type', type = str, default = 'xavier', help = 'the initialization type')
    parser.add_argument('--init_gain', type = float, default = 0.02, help = 'the initialization gain')
    # Dataset parameters
    parser.add_argument('--imgroot', type = str, \
        default = "C:\\Users\\yzzha\\Desktop\\dataset\\ILSVRC2012_val_256\\ILSVRC2012_val_00044958.JPEG", \
            help = 'the base training folder')
    parser.add_argument('--imgsize', type = int, default = 256, help = 'size of image')
    # color scribble parameters
    parser.add_argument('--color_point', type = int, default = 30, help = 'number of color scribbles')
    parser.add_argument('--color_width', type = int, default = 5, help = 'width of each color scribble')
    parser.add_argument('--color_blur_width', type = int, default = 11, help = 'Gaussian blur width of each color scribble')
    opt = parser.parse_args()
    print(opt)

    # ----------------------------------------
    #       Initialize testing dataset
    # ----------------------------------------

    utils.check_path(opt.save_path)

    # Define the dataset
    testset = dataset.ScribbleColorizationValDataset_SingleImg(opt)
    print('The overall number of images equals to %d' % len(testset))

    # Define the dataloader
    dataloader = DataLoader(testset, batch_size = opt.batch_size, pin_memory = True)

    # ----------------------------------------
    #                 Testing
    # ----------------------------------------

    generator = utils.create_generator(opt).cuda()
	
    for i in range(100):

        print('Now the %d-th iteration.' % (i))

        for batch_idx, (grayscale, img, color_scribble, imgname) in enumerate(dataloader):

            # Load and put to cuda
            grayscale = grayscale.cuda()                                    # out: [B, 1, 256, 256]
            img = img.cuda()                                                # out: [B, 3, 256, 256]
            color_scribble = color_scribble.cuda()                          # out: [B, 3, 256, 256]
            print(imgname)
            
            # Generator output
            with torch.no_grad():
                img_out = generator(grayscale, color_scribble)              # out: [B, 3, 256, 256]

            # convert to visible image format
            img = img.cpu().numpy().reshape(3, opt.imgsize, opt.imgsize).transpose(1, 2, 0)
            img = img * 255
            img = img.astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img_out = img_out.detach().cpu().numpy().reshape(3, opt.imgsize, opt.imgsize).transpose(1, 2, 0)
            img_out = img_out * 255
            img_out = img_out.astype(np.uint8)
            img_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)
            
            color_scribble = color_scribble.detach().cpu().numpy().reshape(3, opt.imgsize, opt.imgsize).transpose(1, 2, 0)
            color_scribble = color_scribble * 255
            color_scribble = color_scribble.astype(np.uint8)
            color_scribble = cv2.cvtColor(color_scribble, cv2.COLOR_BGR2RGB)
            
            # show
            '''
            show_img = np.concatenate((img, img_out, color_scribble), axis = 1)
            cv2.imshow('demo', show_img)
            cv2.waitKey(0)
            '''
            
            # save
            print('The saving image name is:')
            save_name = imgname[0].split('\\')[-1][:-5] + '_' + str(i) + '.png'
            save_name = os.path.join(opt.save_path, save_name)
            print(save_name)
            cv2.imwrite(save_name, img_out)
