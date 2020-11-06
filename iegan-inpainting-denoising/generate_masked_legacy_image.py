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
    parser.add_argument('--val_path', type = str, \
        default = "F:\\submitted papers\\my papers\\ACCV, Legacy Photo Editing with Learned Noise Prior\\comparison results\\sample for comparison\\teaser\\legacy_sample_masked_input", \
            help = 'gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--mask_path', type = str, \
        default = "F:\\submitted papers\\my papers\\ACCV, Legacy Photo Editing with Learned Noise Prior\\comparison results\\sample for comparison\\teaser\\legacy_sample_mask", \
            help = 'gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--test_batch_size', type = int, default = 1, help = 'test batch size')
    parser.add_argument('--num_workers', type = int, default = 1, help = 'num of workers')
    # Dataset parameters
    parser.add_argument('--baseroot', type = str, \
        default = "F:\\submitted papers\\my papers\\ACCV, Legacy Photo Editing with Learned Noise Prior\\comparison results\\sample for comparison\\teaser\\legacy_sample_input", \
            help = 'the base training folder for inpainting network')
    parser.add_argument('--maskroot', type = str, default = "./huahen/processed", \
        help = 'the base training folder for inpainting network')
    parser.add_argument('--crop_size', type = int, default = 1760, help = 'size of image')
    parser.add_argument('--mask_size', type = int, default = 512, help = 'size of image')
    parser.add_argument('--noise_aug', type = bool, default = False, help = 'whether add noise to each image')
    parser.add_argument('--noise_level', type = float, default = 0.03, help = 'noise level for each image')
    opt = parser.parse_args()

    # ----------------------------------------
    #                   Test
    # ----------------------------------------
    test_dataset = dataset.Generate_mask_and_masked_image_Legacy_Image_Dataset(opt)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = opt.test_batch_size, shuffle = False, num_workers = opt.num_workers, pin_memory = True)
    utils.check_path(opt.val_path)
    utils.check_path(opt.mask_path)

    # forward
    for i, (masked_img, gt, mask, imgname) in enumerate(test_loader):

        # To device
        print(i, imgname[0])

        # Forward propagation
        # Save
        masked_img = masked_img.clone().data[0, 0, :, :].numpy()
        masked_img = (masked_img * 255.0).astype(np.uint8)
        mask = mask.clone().data[0, 0, :, :].numpy()
        mask = (mask * 255.0).astype(np.uint8)

        save_img_path = os.path.join(opt.val_path, imgname[0])
        cv2.imwrite(save_img_path, masked_img)
        save_img_path = os.path.join(opt.mask_path, imgname[0])
        cv2.imwrite(save_img_path, mask)
        