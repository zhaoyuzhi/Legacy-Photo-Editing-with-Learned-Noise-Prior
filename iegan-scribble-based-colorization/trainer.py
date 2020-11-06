import time
import datetime
import os
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import itertools

import network
import dataset
import utils

def Trainer(opt):
    # ----------------------------------------
    #      Initialize training parameters
    # ----------------------------------------

    # cudnn benchmark accelerates the network
    cudnn.benchmark = opt.cudnn_benchmark

    # Handle multiple GPUs
    gpu_num = torch.cuda.device_count()
    print("There are %d GPUs used" % gpu_num)
    opt.batch_size *= gpu_num
    opt.num_workers *= gpu_num
    print("Batch size is changed to %d" % opt.batch_size)
    print("Number of workers is changed to %d" % opt.num_workers)
    
    # Build path folder
    utils.check_path('./models')
    utils.check_path('./samples')

    # Build networks
    generator = utils.create_generator(opt)

    # To device
    if opt.multi_gpu == True:
        generator = nn.DataParallel(generator)
        generator = generator.cuda()
    else:
        generator = generator.cuda()

    # Loss functions
    L1Loss = nn.L1Loss()

    # Optimizers
    optimizer_g = torch.optim.Adam(generator.parameters(), lr = opt.lr_g, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)
    
    # Learning rate decrease
    def adjust_learning_rate(optimizer, epoch, opt, init_lr):
        """Set the learning rate to the initial LR decayed by "lr_decrease_factor" every "lr_decrease_epoch" epochs"""
        lr = init_lr * (opt.lr_decrease_factor ** (epoch // opt.lr_decrease_epoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    # Save the model if pre_train == True
    def save_model(net, epoch, opt):
        """Save the model at "checkpoint_interval" and its multiple"""
        folder_path = './models'
        model_name = 'scribble_colorization_epoch%d_batchsize%d.pth' % (epoch, opt.batch_size)
        model_path = os.path.join(folder_path, model_name)
        if opt.multi_gpu == True:
            if epoch % opt.checkpoint_interval == 0:
                torch.save(net.module.state_dict(), model_path)
                print('The trained model is successfully saved at epoch %d' % (epoch))
        else:
            if epoch % opt.checkpoint_interval == 0:
                torch.save(net.state_dict(), model_path)
                print('The trained model is successfully saved at epoch %d' % (epoch))
    
    # ----------------------------------------
    #       Initialize training dataset
    # ----------------------------------------

    # Define the dataset
    trainset = dataset.ScribbleColorizationDataset(opt)
    print('The overall number of images equals to %d' % len(trainset))

    # Define the dataloader
    dataloader = DataLoader(trainset, batch_size = opt.batch_size, shuffle = True, num_workers = opt.num_workers, pin_memory = True)
    
    # ----------------------------------------
    #            Training and Testing
    # ----------------------------------------

    # Initialize start time
    prev_time = time.time()

    # Training loop
    for epoch in range(opt.epochs):
        for batch_idx, (grayscale, img, color_scribble, target) in enumerate(dataloader):

            # Load and put to cuda
            grayscale = grayscale.cuda()                                    # out: [B, 1, 256, 256]
            img = img.cuda()                                                # out: [B, 3, 256, 256]
            color_scribble = color_scribble.cuda()                          # out: [B, 3, 256, 256]
            target = target.cuda()

            optimizer_g.zero_grad()

            # forward propagation
            img_out = generator(grayscale, color_scribble)

            ### second stage: jointly denoising and colorization
            # Color L1 Loss
            ColorL1Loss = L1Loss(img_out, img)

            # Compute losses
            loss = ColorL1Loss
            loss.backward()
            optimizer_g.step()

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + batch_idx
            batches_left = opt.epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds = batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            print("\r[Epoch %d/%d] [Batch %d/%d] [Color L1 Loss: %.5f] time_left: %s" %
                ((epoch + 1), opt.epochs, batch_idx, len(dataloader), ColorL1Loss.item(), time_left))

        # Learning rate decrease
        adjust_learning_rate(optimizer_g, (epoch + 1), opt, opt.lr_g)

        # Save the model
        save_model(generator, (epoch + 1), opt)
        utils.sample(grayscale, img, color_scribble, img_out, './samples', (epoch + 1))

def Trainer_GAN(opt):
    # ----------------------------------------
    #      Initialize training parameters
    # ----------------------------------------

    # cudnn benchmark accelerates the network
    cudnn.benchmark = opt.cudnn_benchmark

    # Handle multiple GPUs
    gpu_num = torch.cuda.device_count()
    print("There are %d GPUs used" % gpu_num)
    opt.batch_size *= gpu_num
    opt.num_workers *= gpu_num
    print("Batch size is changed to %d" % opt.batch_size)
    print("Number of workers is changed to %d" % opt.num_workers)
    
    # Build path folder
    utils.check_path('./models')
    utils.check_path('./samples')

    # Build networks
    generator = utils.create_generator(opt)
    discriminator = utils.create_discriminator(opt)
    perceptualnet = utils.create_perceptualnet(opt)

    # To device
    if opt.multi_gpu == True:
        generator = nn.DataParallel(generator)
        discriminator = nn.DataParallel(discriminator)
        perceptualnet = nn.DataParallel(perceptualnet)
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        perceptualnet = perceptualnet.cuda()
    else:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        perceptualnet = perceptualnet.cuda()

    # Loss functions
    L1Loss = nn.L1Loss().cuda()
    MSELoss = nn.MSELoss().cuda()

    # Optimizers
    optimizer_g = torch.optim.Adam(generator.parameters(), lr = opt.lr_g, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr = opt.lr_d, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)

    # Learning rate decrease
    def adjust_learning_rate(optimizer, epoch, opt, init_lr):
        """Set the learning rate to the initial LR decayed by "lr_decrease_factor" every "lr_decrease_epoch" epochs"""
        lr = init_lr * (opt.lr_decrease_factor ** (epoch // opt.lr_decrease_epoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    # Save the model if pre_train == True
    def save_model(net, epoch, opt):
        """Save the model at "checkpoint_interval" and its multiple"""
        folder_path = './models'
        model_name = 'scribble_colorization_GAN_epoch%d_batchsize%d.pth' % (epoch, opt.batch_size)
        model_path = os.path.join(folder_path, model_name)
        if opt.multi_gpu == True:
            if epoch % opt.checkpoint_interval == 0:
                torch.save(net.module.state_dict(), model_path)
                print('The trained model is successfully saved at epoch %d' % (epoch))
        else:
            if epoch % opt.checkpoint_interval == 0:
                torch.save(net.state_dict(), model_path)
                print('The trained model is successfully saved at epoch %d' % (epoch))
    
    # ----------------------------------------
    #       Initialize training dataset
    # ----------------------------------------

    # Define the dataset
    trainset = dataset.ScribbleColorizationDataset(opt)
    print('The overall number of images equals to %d' % len(trainset))

    # Define the dataloader
    dataloader = DataLoader(trainset, batch_size = opt.batch_size, shuffle = True, num_workers = opt.num_workers, pin_memory = True)
    
    # ----------------------------------------
    #            Training and Testing
    # ----------------------------------------

    # Initialize start time
    prev_time = time.time()

    # Tensor type
    Tensor = torch.cuda.FloatTensor

    # Training loop
    for epoch in range(opt.epochs):
        for batch_idx, (grayscale, img, color_scribble, target) in enumerate(dataloader):

            # Load and put to cuda
            grayscale = grayscale.cuda()                                    # out: [B, 1, 256, 256]
            img = img.cuda()                                                # out: [B, 3, 256, 256]
            color_scribble = color_scribble.cuda()                          # out: [B, 3, 256, 256]
            target = target.cuda()

            # LSGAN vectors
            valid = Tensor(np.ones((img.shape[0], 1, 8, 8)))
            fake = Tensor(np.zeros((img.shape[0], 1, 8, 8)))

            # ----------------------------------------
            #           Train Discriminator
            # ----------------------------------------
            optimizer_d.zero_grad()

            # forward propagation
            img_out = generator(grayscale, color_scribble)
			
            ### second stage: jointly denoising and colorization
            # Fake samples
            fake_scalar = discriminator(img_out.detach(), color_scribble)
            # True samples
            true_scalar = discriminator(img, color_scribble)
            # Overall Loss and optimize
            loss_fake = MSELoss(fake_scalar, fake)
            loss_true = MSELoss(true_scalar, valid)
            # Overall Loss and optimize
            loss_D = 0.5 * (loss_fake + loss_true)
            loss_D.backward()

            # ----------------------------------------
            #             Train Generator
            # ----------------------------------------
            optimizer_g.zero_grad()

            # forward propagation
            img_out = generator(grayscale, color_scribble)

            ### second stage: jointly denoising and colorization
            # Color L1 Loss
            ColorL1Loss = L1Loss(img_out, img)
            # GAN Loss
            fake_scalar = discriminator(img_out, color_scribble)
            ColorGAN_Loss = MSELoss(fake_scalar, valid)
            # Get the deep semantic feature maps, and compute Perceptual Loss
            img_featuremaps = perceptualnet(img)                            # feature maps
            img_out_featuremaps = perceptualnet(img_out)
            PerceptualLoss = L1Loss(img_featuremaps, img_out_featuremaps)

            # Compute losses
            loss = opt.lambda_l1 * ColorL1Loss + \
                opt.lambda_perceptual * PerceptualLoss + \
                    opt.lambda_gan * ColorGAN_Loss
            loss.backward()
            optimizer_g.step()

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + batch_idx
            batches_left = opt.epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds = batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            print("\r[Epoch %d/%d] [Batch %d/%d] [Color L1 Loss: %.5f] [Perceptual Loss: %.5f]" %
                ((epoch + 1), opt.epochs, batch_idx, len(dataloader), ColorL1Loss.item(), PerceptualLoss.item()))
            print("\r[D Loss: %.5f] [G Loss: %.5f] time_left: %s" %
                (loss_D.item(), ColorGAN_Loss.item(), time_left))

        # Learning rate decrease
        adjust_learning_rate(optimizer_g, (epoch + 1), opt, opt.lr_g)
        adjust_learning_rate(optimizer_d, (epoch + 1), opt, opt.lr_d)

        # Save the model
        save_model(generator, (epoch + 1), opt)
        utils.sample(grayscale, img, color_scribble, img_out, './samples', (epoch + 1))

'''
def Trainer_GAN(opt):
    # ----------------------------------------
    #      Initialize training parameters
    # ----------------------------------------

    # cudnn benchmark accelerates the network
    cudnn.benchmark = opt.cudnn_benchmark

    # Handle multiple GPUs
    gpu_num = torch.cuda.device_count()
    print("There are %d GPUs used" % gpu_num)
    opt.batch_size *= gpu_num
    opt.num_workers *= gpu_num
    print("Batch size is changed to %d" % opt.batch_size)
    print("Number of workers is changed to %d" % opt.num_workers)
    
    # Build path folder
    utils.check_path('./models')
    utils.check_path('./samples')

    # Build networks
    generator = utils.create_generator(opt)
    discriminator = utils.create_discriminator(opt)
    perceptualnet = utils.create_perceptualnet(opt)
    classificationnet = utils.create_classificationnet(opt)

    # To device
    if opt.multi_gpu == True:
        generator = nn.DataParallel(generator)
        discriminator = nn.DataParallel(discriminator)
        perceptualnet = nn.DataParallel(perceptualnet)
        classificationnet = nn.DataParallel(classificationnet)
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        perceptualnet = perceptualnet.cuda()
        classificationnet = classificationnet.cuda()
    else:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        perceptualnet = perceptualnet.cuda()
        classificationnet = classificationnet.cuda()

    # Loss functions
    L1Loss = nn.L1Loss().cuda()
    MSELoss = nn.MSELoss().cuda()
    CELoss = nn.CrossEntropyLoss().cuda()

    # Optimizers
    optimizer_g = torch.optim.Adam(generator.parameters(), lr = opt.lr_g, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)
    optimizer_d = torch.optim.Adam(generator.parameters(), lr = opt.lr_d, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)

    # Learning rate decrease
    def adjust_learning_rate(optimizer, epoch, opt, init_lr):
        """Set the learning rate to the initial LR decayed by "lr_decrease_factor" every "lr_decrease_epoch" epochs"""
        lr = init_lr * (opt.lr_decrease_factor ** (epoch // opt.lr_decrease_epoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    # Save the model if pre_train == True
    def save_model(net, epoch, opt):
        """Save the model at "checkpoint_interval" and its multiple"""
        folder_path = './models'
        model_name = 'scribble_colorization_GAN_epoch%d_batchsize%d.pth' % (epoch, opt.batch_size)
        model_path = os.path.join(folder_path, model_name)
        if opt.multi_gpu == True:
            if epoch % opt.checkpoint_interval == 0:
                torch.save(net.module.state_dict(), model_path)
                print('The trained model is successfully saved at epoch %d' % (epoch))
        else:
            if epoch % opt.checkpoint_interval == 0:
                torch.save(net.state_dict(), model_path)
                print('The trained model is successfully saved at epoch %d' % (epoch))
    
    # ----------------------------------------
    #       Initialize training dataset
    # ----------------------------------------

    # Define the dataset
    trainset = dataset.ScribbleColorizationDataset(opt)
    print('The overall number of images equals to %d' % len(trainset))

    # Define the dataloader
    dataloader = DataLoader(trainset, batch_size = opt.batch_size, shuffle = True, num_workers = opt.num_workers, pin_memory = True)
    
    # ----------------------------------------
    #            Training and Testing
    # ----------------------------------------

    # Initialize start time
    prev_time = time.time()

    # Tensor type
    Tensor = torch.cuda.FloatTensor

    # Training loop
    for epoch in range(opt.epochs):
        for batch_idx, (grayscale, img, color_scribble, target) in enumerate(dataloader):

            # Load and put to cuda
            grayscale = grayscale.cuda()                                    # out: [B, 1, 256, 256]
            img = img.cuda()                                                # out: [B, 3, 256, 256]
            color_scribble = color_scribble.cuda()                          # out: [B, 3, 256, 256]
            target = target.cuda()

            # LSGAN vectors
            valid = Tensor(np.ones((img.shape[0], 1, 8, 8)))
            fake = Tensor(np.zeros((img.shape[0], 1, 8, 8)))

            # ----------------------------------------
            #           Train Discriminator
            # ----------------------------------------
            optimizer_d.zero_grad()

            # forward propagation
            img_out = generator(grayscale, color_scribble)
			
            ### second stage: jointly denoising and colorization
            # Fake samples
            fake_scalar = discriminator(img_out.detach(), color_scribble)
            # True samples
            true_scalar = discriminator(img, color_scribble)
            # Overall Loss and optimize
            loss_fake = MSELoss(fake_scalar, fake)
            loss_true = MSELoss(true_scalar, valid)
            # Overall Loss and optimize
            loss_D = 0.5 * (loss_fake + loss_true)
            loss_D.backward()

            # ----------------------------------------
            #             Train Generator
            # ----------------------------------------
            optimizer_g.zero_grad()

            # forward propagation
            img_out = generator(grayscale, color_scribble)

            ### second stage: jointly denoising and colorization
            # Color L1 Loss
            ColorL1Loss = L1Loss(img_out, img)
            # GAN Loss
            fake_scalar = discriminator(img_out, color_scribble)
            ColorGAN_Loss = MSELoss(fake_scalar, valid)
            # Get the deep semantic feature maps, and compute Perceptual Loss
            img_featuremaps = perceptualnet(img)                            # feature maps
            img_out_featuremaps = perceptualnet(img_out)
            PerceptualLoss = L1Loss(img_featuremaps, img_out_featuremaps)
            # Classification Loss
            output = classificationnet(img_out)
            ClassificationLoss = CELoss(output, target)

            # Compute losses
            loss = opt.lambda_l1 * ColorL1Loss + \
                opt.lambda_perceptual * PerceptualLoss + \
                    opt.lambda_gan * ColorGAN_Loss + \
                        opt.lambda_classification * ClassificationLoss
            loss.backward()
            optimizer_g.step()

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + batch_idx
            batches_left = opt.epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds = batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            print("\r[Epoch %d/%d] [Batch %d/%d] [Color L1 Loss: %.5f] [Perceptual Loss: %.5f]" %
                ((epoch + 1), opt.epochs, batch_idx, len(dataloader), ColorL1Loss.item(), PerceptualLoss.item()))
            print("\r[CrossEntropy Loss: %.5f] [D Loss: %.5f] [G Loss: %.5f] time_left: %s" %
                (ClassificationLoss.item(), loss_D.item(), ColorGAN_Loss.item(), time_left))

        # Learning rate decrease
        adjust_learning_rate(optimizer_g, (epoch + 1), opt, opt.lr_g)
        adjust_learning_rate(optimizer_d, (epoch + 1), opt, opt.lr_d)

        # Save the model
        save_model(generator, (epoch + 1), opt)
        utils.sample(grayscale, img, color_scribble, img_out, './samples', (epoch + 1))
'''