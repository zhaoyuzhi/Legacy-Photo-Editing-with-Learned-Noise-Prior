import argparse
import os
import torch
import numpy as np
import cv2

import utils
import dataset

if __name__ == "__main__":
    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    # General parameters
    parser.add_argument('--pre_train', type = bool, default = True, help = 'the type of GAN for training')
    parser.add_argument('--finetune_path', type = str, \
        default = "./models/GrayInpainting_epoch5_batchsize16_noise0.03.pth", \
            help = 'the load name of models')
    parser.add_argument('--val_path', type = str, \
        default = "F:\\submitted papers\\my papers\\ACCV, Legacy Photo Editing with Learned Noise Prior\\comparison results\\input\\", \
            help = 'gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    # Network parameters
    parser.add_argument('--in_channels', type = int, default = 1, help = 'input RGB image')
    parser.add_argument('--out_channels', type = int, default = 1, help = 'output RGB image')
    parser.add_argument('--mask_channels', type = int, default = 1, help = 'input mask')
    parser.add_argument('--latent_channels', type = int, default = 64, help = 'latent channels')
    parser.add_argument('--pad', type = str, default = 'reflect', help = 'the padding type')
    parser.add_argument('--activ_g', type = str, default = 'lrelu', help = 'the activation type')
    parser.add_argument('--activ_d', type = str, default = 'lrelu', help = 'the activation type')
    parser.add_argument('--norm_g', type = str, default = 'in', help = 'normalization type')
    parser.add_argument('--norm_d', type = str, default = 'bn', help = 'normalization type')
    # Dataset parameters
    parser.add_argument('--baseroot', type = str, \
        # old image photo           old image photo pesudo noisy
        default = "F:\\submitted papers\\my papers\\ACCV, Legacy Photo Editing with Learned Noise Prior\\comparison results\\input\\ILSVRC2012_val_256_masked_image\\ILSVRC2012_val_00000006.JPEG", \
            help = 'the base training folder for inpainting network')
    parser.add_argument('--resize', type = int, default = 256, help = 'size of image')
    opt = parser.parse_args()

    # ----------------------------------------
    #                   Test
    # ----------------------------------------
    # Initialize
    generator = utils.create_generator(opt).cuda()
    utils.check_path(opt.val_path)

    # data processing
    img = cv2.imread(opt.baseroot, flags = cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (opt.resize, opt.resize))
    img = img.astype(np.float64) / 255
    img = torch.from_numpy(img.astype(np.float32)).unsqueeze(0).unsqueeze(0).contiguous()

    # forward
    img = img.cuda()
    with torch.no_grad():
        img = generator(img)

    img = img.clone().data.permute(0, 2, 3, 1)[0, :, :, 0].cpu().numpy()
    img = (img * 255.0).astype(np.uint8)
    save_name = opt.baseroot.split('\\')[-1].split('.')[0] + '.png'
    save_img_path = os.path.join(opt.val_path, save_name)
    cv2.imwrite(save_img_path, img)
