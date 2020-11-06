import argparse
import os

import trainer

if __name__ == "__main__":
    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    # Pre-train, saving, and loading parameters
    parser.add_argument('--save_path', type = str, default = './models', help = 'saving path that is a folder')
    parser.add_argument('--sample_path', type = str, default = './samples', help = 'training samples path that is a folder')
    parser.add_argument('--save_mode', type = str, default = 'iter', help = 'saving mode, and by_epoch saving is recommended')
    parser.add_argument('--save_by_epoch', type = int, default = 1, help = 'interval between model checkpoints (by epochs)')
    parser.add_argument('--save_by_iter', type = int, default = 5000, help = 'interval between model checkpoints (by iterations)')
    parser.add_argument('--wavelet_name_ab', type = str, default = '', help = 'wavelet_name_ab')
    parser.add_argument('--wavelet_name_ba', type = str, default = '', help = 'wavelet_name_ba')
    # GPU parameters
    parser.add_argument('--multi_gpu', type = bool, default = False, help = 'True for more than 1 GPU')
    parser.add_argument('--gpu_ids', type = str, default = '0, 1, 2, 3', help = 'gpu_ids: e.g. 0  0,1  0,1,2  use -1 for CPU')
    parser.add_argument('--cudnn_benchmark', type = bool, default = True, help = 'True for unchanged input data type')
    # Training parameters
    parser.add_argument('--epochs', type = int, default = 300, help = 'number of epochs of training')
    parser.add_argument('--batch_size', type = int, default = 1, help = 'size of the batches')
    parser.add_argument('--lr_g', type = float, default = 0.0002, help = 'Adam: learning rate for G')
    parser.add_argument('--lr_d', type = float, default = 0.0002, help = 'Adam: learning rate for D')
    parser.add_argument('--b1', type = float, default = 0.5, help = 'Adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type = float, default = 0.999, help = 'Adam: decay of second order momentum of gradient')
    parser.add_argument('--weight_decay', type = float, default = 0, help = 'Adam: weight decay')
    parser.add_argument('--lr_decrease_mode', type = str, default = 'iter', help = 'lr decrease mode')
    parser.add_argument('--lr_decrease_epoch', type = int, default = 1, help = 'lr decrease at certain epoch and its multiple')
    parser.add_argument('--lr_decrease_iter', type = int, default = 10000, help = 'lr decrease at certain epoch and its multiple')
    parser.add_argument('--lr_decrease_factor', type = float, default = 0.5, help = 'lr decrease factor, for classification default 0.1')
    parser.add_argument('--lambda_wavelet_l1', type = float, default = 0.8, help = 'coefficient for Wavelet low-pass L1 Loss')
    parser.add_argument('--lambda_wavelet_gan', type = float, default = 1, help = 'coefficient for high-pass GAN Loss')
    parser.add_argument('--lambda_cycle', type = float, default = 10, help = 'the parameter of L1Loss')
    parser.add_argument('--lambda_identity', type = float, default = 0, help = 'the parameter of perceptual loss')
    parser.add_argument('--num_workers', type = int, default = 8, help = 'number of cpu threads to use during batch generation')
    # Initialization parameters
    parser.add_argument('--in_channels', type = int, default = 1, help = '1 for colorization, 3 for other tasks')
    parser.add_argument('--out_channels', type = int, default = 1, help = '2 for colorization, 3 for other tasks')
    parser.add_argument('--start_channels', type = int, default = 64, help = 'start channels for the main stream of generator')
    parser.add_argument('--wavelet_channels', type = int, default = 3, help = 'wavelet channels')
    parser.add_argument('--pad', type = str, default = 'reflect', help = 'pad type of networks')
    parser.add_argument('--activ_g', type = str, default = 'lrelu', help = 'activation type of networks for generator')
    parser.add_argument('--activ_d', type = str, default = 'lrelu', help = 'activation type of networks for discriminator')
    parser.add_argument('--norm_g', type = str, default = 'none', help = 'normalization type of networks for generator')
    parser.add_argument('--norm_d', type = str, default = 'none', help = 'normalization type of networks for discriminator')
    parser.add_argument('--init_type', type = str, default = 'xavier', help = 'initialization type of networks')
    parser.add_argument('--init_gain', type = float, default = 0.02, help = 'initialization gain of networks')
    # Dataset parameters
    parser.add_argument('--baseroot_A', type = str, default = 'C:\\Users\\yzzha\\Desktop\\dataset\\ILSVRC2012_val_256', help = 'clean images set path')
    parser.add_argument('--baseroot_B', type = str, default = 'E:\\dataset, my paper related\\Legacy Image dataset\\old image photo', help = 'noisy images set path')
    parser.add_argument('--crop_size', type = int, default = 256, help = 'crop patch size for HR patch')
    opt = parser.parse_args()
    
    # ----------------------------------------
    #                 Trainer
    # ----------------------------------------
    trainer.WaveletGAN_Trainer(opt)
