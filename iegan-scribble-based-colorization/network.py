import torch
import torch.nn as nn

from network_module import *

def weights_init(net, init_type = 'kaiming', init_gain = 0.02):
    """Initialize network weights.
    Parameters:
        net (network)       -- network to be initialized
        init_type (str)     -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_var (float)    -- scaling factor for normal, xavier and orthogonal.
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain = init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a = 0, mode = 'fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain = init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            init.normal_(m.weight, 0, 0.01)
            init.constant_(m.bias, 0)

    # Apply the initialization function <init_func>
    net.apply(init_func)

#-----------------------------------------------
#                   Generator
#-----------------------------------------------
# Input: masked image + mask
# Output: filled image
class ScribbleColorNet(nn.Module):
    def __init__(self, opt):
        super(ScribbleColorNet, self).__init__()
        self.down1 = Conv2dLayer(opt.in_channels + opt.scribble_channels, opt.start_channels, 7, 1, 3, pad_type = opt.pad, activation = opt.activ_g, norm = 'none')
        self.down2 = Conv2dLayer(opt.start_channels, opt.start_channels * 2, 4, 2, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.down3 = Conv2dLayer(opt.start_channels * 2, opt.start_channels * 4, 4, 2, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.down4 = Conv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 4, 2, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.down5 = Conv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 4, 2, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.down6 = Conv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 4, 2, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.down7 = Conv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 4, 2, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.up1 = TransposeGatedConv2d(opt.start_channels * 4, opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.up2 = TransposeGatedConv2d(opt.start_channels * 8, opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.up3 = TransposeGatedConv2d(opt.start_channels * 8, opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.up4 = TransposeGatedConv2d(opt.start_channels * 8, opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.up5 = TransposeGatedConv2d(opt.start_channels * 8, opt.start_channels * 2, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.up6 = TransposeGatedConv2d(opt.start_channels * 4, opt.start_channels, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.up7 = Conv2dLayer(opt.start_channels * 2, opt.out_channels, 3, 1, 1, pad_type = opt.pad, activation = 'sigmoid', norm = 'none')

    def forward(self, img, scribble):
        # the input x should contain 4 channels because it is a combination of input grayscale image and scribble
        x = torch.cat((img, scribble), 1)
        down1 = self.down1(x)                                       # out: batch * 64 * 256 * 256
        down2 = self.down2(down1)                                   # out: batch * 128 * 128 * 128
        down3 = self.down3(down2)                                   # out: batch * 256 * 64 * 64
        down4 = self.down4(down3)                                   # out: batch * 256 * 32 * 32
        down5 = self.down5(down4)                                   # out: batch * 256 * 16 * 16
        down6 = self.down6(down5)                                   # out: batch * 256 * 8 * 8
        down7 = self.down7(down6)                                   # out: batch * 256 * 4 * 4
        up1 = self.up1(down7)                                       # out: batch * 256 * 8 * 8
        up1 = torch.cat((up1, down6), 1)                            # out: batch * 512 * 8 * 8
        up2 = self.up2(up1)                                         # out: batch * 256 * 16 * 16
        up2 = torch.cat((up2, down5), 1)                            # out: batch * 512 * 16 * 16
        up3 = self.up3(up2)                                         # out: batch * 256 * 32 * 32
        up3 = torch.cat((up3, down4), 1)                            # out: batch * 512 * 32 * 32
        up4 = self.up4(up3)                                         # out: batch * 256 * 64 * 64
        up4 = torch.cat((up4, down3), 1)                            # out: batch * 512 * 64 * 64
        up5 = self.up5(up4)                                         # out: batch * 128 * 128 * 128
        up5 = torch.cat((up5, down2), 1)                            # out: batch * 256 * 128 * 128
        up6 = self.up6(up5)                                         # out: batch * 64 * 256 * 256
        up6 = torch.cat((up6, down1), 1)                            # out: batch * 128 * 256 * 256
        up7 = self.up7(up6)                                         # out: batch * 3 * 256 * 256
        return up7

'''
class SGN(nn.Module):
    def __init__(self, opt):
        super(SGN, self).__init__()
        # Top subnetwork, K = 3
        self.top1 = Conv2dLayer((opt.in_channels + opt.scribble_channels) * (4 ** 3), opt.start_channels * (2 ** 3), 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.top2 = ResConv2dLayer(opt.start_channels * (2 ** 3), 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.top3 = Conv2dLayer(opt.start_channels * (2 ** 3), opt.start_channels * (2 ** 3), 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        # Middle subnetwork, K = 2
        self.mid1 = Conv2dLayer((opt.in_channels + opt.scribble_channels) * (4 ** 2), opt.start_channels * (2 ** 2), 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.mid2 = Conv2dLayer(int(opt.start_channels * (2 ** 2 + 2 ** 3 / 4)), opt.start_channels * (2 ** 2), 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.mid3 = ResConv2dLayer(opt.start_channels * (2 ** 2), 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.mid4 = Conv2dLayer(opt.start_channels * (2 ** 2), opt.start_channels * (2 ** 2), 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        # Bottom subnetwork, K = 1
        self.bot1 = Conv2dLayer((opt.in_channels + opt.scribble_channels) * (4 ** 1), opt.start_channels * (2 ** 1), 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.bot2 = Conv2dLayer(int(opt.start_channels * (2 ** 1 + 2 ** 2 / 4)), opt.start_channels * (2 ** 1), 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.bot3 = ResConv2dLayer(opt.start_channels * (2 ** 1), 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.bot4 = Conv2dLayer(opt.start_channels * (2 ** 1), opt.start_channels * (2 ** 1), 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        # Mainstream
        self.main1 = Conv2dLayer((opt.in_channels + opt.scribble_channels), opt.start_channels, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.main2 = Conv2dLayer(int(opt.start_channels * (2 ** 0 + 2 ** 1 / 4)), opt.start_channels, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.main3_1 = ResConv2dLayer(opt.start_channels, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.main3_2 = ResConv2dLayer(opt.start_channels, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.main3_3 = ResConv2dLayer(opt.start_channels, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.main3_4 = Conv2dLayer(opt.start_channels, opt.start_channels, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.main4 = Conv2dLayer(opt.start_channels, opt.out_channels, 3, 1, 1, pad_type = opt.pad, activation = 'none', norm = 'none')

    def forward(self, img, scribble):
        # the input x should contain 4 channels because it is a combination of input image and mask
        x = torch.cat((img, scribble), 1)
        # PixelUnShuffle                                        input: batch * 3 * 256 * 256
        x1 = PixelUnShuffle.pixel_unshuffle(x, 2)               # out: batch * 12 * 128 * 128
        x2 = PixelUnShuffle.pixel_unshuffle(x1, 2)              # out: batch * 48 * 64 * 64
        x3 = PixelUnShuffle.pixel_unshuffle(x2, 2)              # out: batch * 192 * 32 * 32
        # Top subnetwork                                        suppose the start_channels = 32
        x3 = self.top1(x3)                                      # out: batch * 256 * 32 * 32
        x3 = self.top2(x3)                                      # out: batch * 256 * 32 * 32
        x3 = self.top3(x3)                                      # out: batch * 256 * 32 * 32
        x3 = F.pixel_shuffle(x3, 2)                             # out: batch * 64 * 64 * 64, ready to be concatenated
        # Middle subnetwork
        x2 = self.mid1(x2)                                      # out: batch * 128 * 64 * 64
        x2 = torch.cat((x2, x3), 1)                             # out: batch * (128 + 64) * 64 * 64
        x2 = self.mid2(x2)                                      # out: batch * 128 * 64 * 64
        x2 = self.mid3(x2)                                      # out: batch * 128 * 64 * 64
        x2 = self.mid4(x2)                                      # out: batch * 128 * 64 * 64
        x2 = F.pixel_shuffle(x2, 2)                             # out: batch * 32 * 128 * 128, ready to be concatenated
        # Bottom subnetwork
        x1 = self.bot1(x1)                                      # out: batch * 64 * 128 * 128
        x1 = torch.cat((x1, x2), 1)                             # out: batch * (64 + 32) * 128 * 128
        x1 = self.bot2(x1)                                      # out: batch * 64 * 128 * 128
        x1 = self.bot3(x1)                                      # out: batch * 64 * 128 * 128
        x1 = self.bot4(x1)                                      # out: batch * 64 * 128 * 128
        x1 = F.pixel_shuffle(x1, 2)                             # out: batch * 16 * 256 * 256, ready to be concatenated
        # U-Net generator with skip connections from encoder to decoder
        x = self.main1(x)                                       # out: batch * 32 * 256 * 256
        x = torch.cat((x, x1), 1)                               # out: batch * (32 + 16) * 256 * 256
        x = self.main2(x)                                       # out: batch * 32 * 256 * 256
        x = self.main3_1(x)                                     # out: batch * 32 * 256 * 256
        x = self.main3_2(x)                                     # out: batch * 32 * 256 * 256
        x = self.main3_3(x)                                     # out: batch * 32 * 256 * 256
        x = self.main3_4(x)                                     # out: batch * 32 * 256 * 256
        x = self.main4(x)                                       # out: batch * 3 * 256 * 256
        return x
'''
#-----------------------------------------------
#                  Discriminator
#-----------------------------------------------
# Input: generated image / ground truth and mask
# Output: patch based region, we set 30 * 30
class PatchDiscriminator(nn.Module):
    def __init__(self, opt):
        super(PatchDiscriminator, self).__init__()
        # Down sampling
        self.block1 = Conv2dLayer(opt.out_channels + opt.scribble_channels, opt.start_channels, 7, 1, 3, pad_type = opt.pad, activation = opt.activ_d, norm = 'none', sn = True)
        self.block2 = Conv2dLayer(opt.start_channels, opt.start_channels * 2, 4, 2, 1, pad_type = opt.pad, activation = opt.activ_d, norm = opt.norm_d, sn = True)
        self.block3 = Conv2dLayer(opt.start_channels * 2, opt.start_channels * 4, 4, 2, 1, pad_type = opt.pad, activation = opt.activ_d, norm = opt.norm_d, sn = True)
        self.block4 = Conv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 4, 2, 1, pad_type = opt.pad, activation = opt.activ_d, norm = opt.norm_d, sn = True)
        self.block5 = Conv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 4, 2, 1, pad_type = opt.pad, activation = opt.activ_d, norm = opt.norm_d, sn = True)
        self.block6 = Conv2dLayer(opt.start_channels * 4, 1, 4, 2, 1, pad_type = opt.pad, activation = 'none', norm = 'none', sn = True)
        
    def forward(self, img, scribble):
        # the input x should contain 6 channels because it is a combination of recon image and mask
        x = torch.cat((img, scribble), 1)
        x = self.block1(x)                                      # out: batch * 64 * 256 * 256
        x = self.block2(x)                                      # out: batch * 128 * 128 * 128
        x = self.block3(x)                                      # out: batch * 256 * 64 * 64
        x = self.block4(x)                                      # out: batch * 256 * 32 * 32
        x = self.block5(x)                                      # out: batch * 256 * 16 * 16
        x = self.block6(x)                                      # out: batch * 1 * 8 * 8
        return x

# ----------------------------------------
#            Perceptual Network
# ----------------------------------------
# VGG-16 conv4_3 features
class PerceptualNet(nn.Module):
    def __init__(self):
        super(PerceptualNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(512, 512, 3, 1, 1)
        )

    def forward(self, x):
        x = self.features(x)
        return x
