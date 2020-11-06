import os
import time
import datetime
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.autograd as autograd
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from pytorch_wavelets import DWTForward, DWTInverse
from tensorboardX import SummaryWriter

import dataset
import utils

def WaveletGAN_Trainer(opt):
    # ----------------------------------------
    #       Network training parameters
    # ----------------------------------------

    # cudnn benchmark
    cudnn.benchmark = opt.cudnn_benchmark

    # configurations
    save_folder = opt.save_path
    sample_folder = opt.sample_path
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if not os.path.exists(sample_folder):
        os.makedirs(sample_folder)

    # Loss functions
    criterion_L1 = torch.nn.L1Loss().cuda()

    # Initialize Generator
    # A is for grayscale image
    # B is for color RGB image
    G_AB, G_BA = utils.create_wavelet_generator(opt)
    D_A, D_B = utils.create_wavelet_discriminator(opt)
    dwt = DWTForward(J = 1, wave = 'db1')
    idwt = DWTInverse(wave = 'db1')

    # To device
    if opt.multi_gpu:
        G_AB = nn.DataParallel(G_AB)
        G_AB = G_AB.cuda()
        G_BA = nn.DataParallel(G_BA)
        G_BA = G_BA.cuda()
        D_A = nn.DataParallel(D_A)
        D_A = D_A.cuda()
        D_B = nn.DataParallel(D_B)
        D_B = D_B.cuda()
        dwt = nn.DataParallel(dwt)
        dwt = dwt.cuda()
        idwt = nn.DataParallel(idwt)
        idwt = idwt.cuda()
    else:
        G_AB = G_AB.cuda()
        G_BA = G_BA.cuda()
        D_A = D_A.cuda()
        D_B = D_B.cuda()
        dwt = dwt.cuda()
        idwt = idwt.cuda()

    # Optimizers
    optimizer_G = torch.optim.Adam(
        itertools.chain(G_AB.parameters(), G_BA.parameters()), lr = opt.lr_g, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay
    )
    optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr = opt.lr_d, betas = (opt.b1, opt.b2))
    optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr = opt.lr_d, betas = (opt.b1, opt.b2))
    
    # Learning rate decrease
    def adjust_learning_rate(opt, epoch, iteration, optimizer):
        # Set the learning rate to the initial LR decayed by "lr_decrease_factor" every "lr_decrease_epoch" epochs
        if opt.lr_decrease_mode == 'epoch':
            lr = opt.lr_g * (opt.lr_decrease_factor ** (epoch // opt.lr_decrease_epoch))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        if opt.lr_decrease_mode == 'iter':
            lr = opt.lr_g * (opt.lr_decrease_factor ** (iteration // opt.lr_decrease_iter))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    
    # Save the model if pre_train == True
    def save_model(opt, epoch, iteration, len_dataset, G_AB, G_BA):
        """Save the model at "checkpoint_interval" and its multiple"""
        # Define the name of trained model
        if opt.save_mode == 'epoch':
            model_name_AB = 'WaveletGAN_AB_epoch%d_bs%d.pth' % (epoch, opt.batch_size)
            model_name_BA = 'WaveletGAN_BA_epoch%d_bs%d.pth' % (epoch, opt.batch_size)
        if opt.save_mode == 'iter':
            model_name_AB = 'WaveletGAN_AB_iter%d_bs%d.pth' % (iteration, opt.batch_size)
            model_name_BA = 'WaveletGAN_BA_iter%d_bs%d.pth' % (iteration, opt.batch_size)
        save_model_path_AB = os.path.join(opt.save_path, model_name_AB)
        save_model_path_BA = os.path.join(opt.save_path, model_name_BA)
        if opt.multi_gpu == True:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    torch.save(G_AB.module.state_dict(), save_model_path_AB)
                    torch.save(G_BA.module.state_dict(), save_model_path_BA)
                    print('The trained model is successfully saved at epoch %d' % (epoch))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    torch.save(G_AB.module.state_dict(), save_model_path_AB)
                    torch.save(G_BA.module.state_dict(), save_model_path_BA)
                    print('The trained model is successfully saved at iteration %d' % (iteration))
        else:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    torch.save(G_AB.state_dict(), save_model_path_AB)
                    torch.save(G_BA.state_dict(), save_model_path_BA)
                    print('The trained model is successfully saved at epoch %d' % (epoch))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    torch.save(G_AB.state_dict(), save_model_path_AB)
                    torch.save(G_BA.state_dict(), save_model_path_BA)
                    print('The trained model is successfully saved at iteration %d' % (iteration))
    
    # ----------------------------------------
    #             Network dataset
    # ----------------------------------------

    # Define the dataset
    trainset = dataset.DomainTransferDataset(opt)
    print('The overall number of images:', len(trainset))

    # Define the dataloader
    dataloader = DataLoader(trainset, batch_size = opt.batch_size, shuffle = True, num_workers = opt.num_workers, pin_memory = True)
    
    # ----------------------------------------
    #                 Training
    # ----------------------------------------

    # Count start time
    prev_time = time.time()

    # For loop training
    for epoch in range(opt.epochs):
        for i, (true_A, true_B) in enumerate(dataloader):

            # To device
            # A is for clean image
            # B is for noisy image
            true_A = true_A.cuda()
            true_B = true_B.cuda()

            ### Train Generator
            optimizer_G.zero_grad()
            optimizer_D_A.zero_grad()
            optimizer_D_B.zero_grad()

            ###--------------------------------------- G_AB part ---------------------------------------
            # Forward
            pred_A = G_AB(true_A)

            # Discrite Wavelet Transformation
            dwt_true_A_l, dwt_true_A_h = dwt(true_A)
            dwt_pred_A_l, dwt_pred_A_h = dwt(pred_A)
            low_true_A = idwt((dwt_true_A_l, [torch.zeros(1, 1, 3, true_A.shape[2] // 2, true_A.shape[3] // 2).cuda()]))
            low_pred_A = idwt((dwt_pred_A_l, [torch.zeros(1, 1, 3, true_A.shape[2] // 2, true_A.shape[3] // 2).cuda()]))
            #high_pred_A = idwt((torch.ones(1, 1, true_A.shape[2] // 2, true_A.shape[3] // 2).cuda() / 1, dwt_pred_A_h))
            high_pred_A = torch.cat((dwt_pred_A_h[0][:, :, 0, :, :], dwt_pred_A_h[0][:, :, 1, :, :], dwt_pred_A_h[0][:, :, 2, :, :]), 1)

            # Low-frequency part
            #loss_l_AB = criterion_L1(low_pred_A, low_true_A)
            loss_l_AB = criterion_L1(dwt_pred_A_l, dwt_true_A_l)

            # High-frequency part
            fake_scalar = D_B(high_pred_A)
            loss_h_AB = - torch.mean(fake_scalar)

            # Overall Loss
            loss_G_AB = opt.lambda_wavelet_l1 * loss_l_AB + opt.lambda_wavelet_gan * loss_h_AB

            ### Train Discriminator
            # Discrite Wavelet Transformation
            dwt_pred_A_l, dwt_pred_A_h = dwt(pred_A)
            dwt_true_B_l, dwt_true_B_h = dwt(true_B)
            #high_pred_A = idwt((torch.ones(1, 1, true_A.shape[2] // 2, true_A.shape[3] // 2).cuda() / 1, dwt_pred_A_h))
            #high_true_B = idwt((torch.ones(1, 1, true_A.shape[2] // 2, true_A.shape[3] // 2).cuda() / 1, dwt_true_B_h))
            high_pred_A = torch.cat((dwt_pred_A_h[0][:, :, 0, :, :], dwt_pred_A_h[0][:, :, 1, :, :], dwt_pred_A_h[0][:, :, 2, :, :]), 1)
            high_true_B = torch.cat((dwt_true_B_h[0][:, :, 0, :, :], dwt_true_B_h[0][:, :, 1, :, :], dwt_true_B_h[0][:, :, 2, :, :]), 1)

            # GAN loss
            fake_scalar = D_B(high_pred_A.detach())
            true_scalar = D_B(high_true_B)
            
            # Overall Loss
            loss_D_A = torch.mean(fake_scalar) - torch.mean(true_scalar)
            ###--------------------------------------- G_AB part ---------------------------------------
            
            ###--------------------------------------- G_BA part ---------------------------------------
            # Forward
            pred_B = G_BA(true_B)

            # Discrite Wavelet Transformation
            dwt_true_B_l, dwt_true_B_h = dwt(true_B)
            dwt_pred_B_l, dwt_pred_B_h = dwt(pred_B)
            low_true_B = idwt((dwt_true_B_l, [torch.zeros(1, 1, 3, true_A.shape[2] // 2, true_A.shape[3] // 2).cuda()]))
            low_pred_B = idwt((dwt_pred_B_l, [torch.zeros(1, 1, 3, true_A.shape[2] // 2, true_A.shape[3] // 2).cuda()]))
            #high_pred_B = idwt((torch.ones(1, 1, true_A.shape[2] // 2, true_A.shape[3] // 2).cuda() / 1, dwt_pred_B_h))
            high_pred_B = torch.cat((dwt_pred_B_h[0][:, :, 0, :, :], dwt_pred_B_h[0][:, :, 1, :, :], dwt_pred_B_h[0][:, :, 2, :, :]), 1)

            # Low-frequency part
            #loss_l_BA = criterion_L1(low_pred_B, low_true_B)
            loss_l_BA = criterion_L1(dwt_pred_B_l, dwt_true_B_l)

            # High-frequency part
            fake_scalar = D_A(high_pred_B)
            loss_h_BA = - torch.mean(fake_scalar)

            # Overall Loss
            loss_G_BA = opt.lambda_wavelet_l1 * loss_l_BA + opt.lambda_wavelet_gan * loss_h_BA

            ### Train Discriminator
            # Discrite Wavelet Transformation
            dwt_pred_B_l, dwt_pred_B_h = dwt(pred_B)
            dwt_true_A_l, dwt_true_A_h = dwt(true_A)
            #high_pred_B = idwt((torch.ones(1, 1, true_A.shape[2] // 2, true_A.shape[3] // 2).cuda() / 1, dwt_pred_B_h))
            #high_true_A = idwt((torch.ones(1, 1, true_A.shape[2] // 2, true_A.shape[3] // 2).cuda() / 1, dwt_true_A_h))
            high_pred_B = torch.cat((dwt_pred_B_h[0][:, :, 0, :, :], dwt_pred_B_h[0][:, :, 1, :, :], dwt_pred_B_h[0][:, :, 2, :, :]), 1)
            high_true_A = torch.cat((dwt_true_A_h[0][:, :, 0, :, :], dwt_true_A_h[0][:, :, 1, :, :], dwt_true_A_h[0][:, :, 2, :, :]), 1)

            # GAN loss
            fake_scalar = D_A(high_pred_B.detach())
            true_scalar = D_A(high_true_A)
            
            # Overall Loss
            loss_D_B = torch.mean(fake_scalar) - torch.mean(true_scalar)
            ###--------------------------------------- G_BA part ---------------------------------------

            ### Cycle consistency
            # Indentity Loss
            loss_indentity_A = criterion_L1(G_BA(true_A), true_A)
            loss_indentity_B = criterion_L1(G_AB(true_B), true_B)
            loss_indentity = (loss_indentity_A + loss_indentity_B) / 2

            # Cycle-consistency Loss
            recon_A = G_BA(pred_A)
            loss_cycle_A = criterion_L1(recon_A, true_A)
            recon_B = G_AB(pred_B)
            loss_cycle_B = criterion_L1(recon_B, true_B)
            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

            # Overall Loss and optimize
            loss = loss_G_AB + loss_G_BA + opt.lambda_cycle * loss_cycle + opt.lambda_identity * loss_indentity
            loss.backward()
            optimizer_G.step()
            loss_D_A.backward()
            optimizer_D_A.step()
            loss_D_B.backward()
            optimizer_D_B.step()

            # Determine approximate time left
            iters_done = epoch * len(dataloader) + i
            iters_left = opt.epochs * len(dataloader) - iters_done
            time_left = datetime.timedelta(seconds = iters_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            print("\r[Epoch %d/%d] [Batch %d/%d] Time_left: %s [loss_cycle: %.4f] [loss_indentity: %.4f]" %
                ((epoch + 1), opt.epochs, i, len(dataloader), time_left, loss_cycle.item(), loss_indentity.item()))
            print("\r[loss_l_AB: %.4f] [loss_h_AB(G): %.4f] [loss_h_AB(D): %.4f]" %
                (loss_l_AB.item(), loss_h_AB.item(), loss_D_A.item()))
            print("\r[loss_l_BA: %.4f] [loss_h_BA(G): %.4f] [loss_h_BA(D): %.4f]" %
                (loss_l_BA.item(), loss_h_BA.item(), loss_D_B.item()))

            # Save model at certain epochs or iterations
            save_model(opt, (epoch + 1), (iters_done + 1), len(dataloader), G_AB, G_BA)

            # Learning rate decrease at certain epochs
            adjust_learning_rate(opt, (epoch + 1), (iters_done + 1), optimizer_G)
            adjust_learning_rate(opt, (epoch + 1), (iters_done + 1), optimizer_D_A)
            adjust_learning_rate(opt, (epoch + 1), (iters_done + 1), optimizer_D_B)

            ### Sample data every 1000 iterations
            if (iters_done + 1) % 1000 == 0:
                img_list = [low_pred_A, low_true_A]
                name_list = ['pred', 'gt']
                utils.save_sample_png(sample_folder = sample_folder, sample_name = 'train_iter%d' % (iters_done + 1), img_list = img_list, name_list = name_list, pixel_max_cnt = 255)
