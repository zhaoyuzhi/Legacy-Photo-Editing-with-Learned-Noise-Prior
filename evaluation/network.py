# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 12:04:52 2018

@author: yzzhao2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import torch.utils.model_zoo as model_zoo

##############################
#           VGG16
##############################

vgg16 = tv.models.vgg16(pretrained = True)

# fine-tune classification layers
class VGG16(nn.Module):
    def __init__(self, vgg16net = vgg16, num_classes = 1000, init_weights = False):
        super(VGG16, self).__init__()
        self.features = vgg16net.features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
