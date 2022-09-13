#! python3
# encoding: utf-8
"""
@author:    Yize Wang
@contact:    wangyize@hust.edu.cn
@file:    unet_model.py
@time:   2022/3/16 16:25
@description:    assemble functions to complete UNet
"""

import torch.nn as nn
from .unet_funcs import DoubleConv, DownSampling, UpSampling, OutConv


class UNet(nn.Module):
    """ assemble all unet parts to complete the network """

    def __init__(self, n_channels, n_classes):
        """
            n_channels: number of input channels, rbg is 3, grey is 1
            n_classes: number of classes, one class with background is 1, two classes is one, n > 2 classes is n
        """

        super(UNet, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes

        assert self.n_classes == 1, "Current codes only support binary classification!"

        self.input_layer = DoubleConv(n_channels, 64)
        self.down1 = DownSampling(64, 128)
        self.down2 = DownSampling(128, 256)
        self.down3 = DownSampling(256, 512)
        self.down4 = DownSampling(512, 1024)

        self.up1 = UpSampling(1024, 512)
        self.up2 = UpSampling(512, 256)
        self.up3 = UpSampling(256, 128)
        self.up4 = UpSampling(128, 64)
        self.output_layer = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.input_layer(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.output_layer(x)
