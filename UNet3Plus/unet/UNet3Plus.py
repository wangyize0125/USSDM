#! python3
# encoding: utf-8
"""
@author:    Yize Wang
@contact:    wangyize@hust.edu.cn
@file:    UNet3Plus.py
@time:   2022/3/20 9:31
@description:    
"""

import torch
import torch.nn as nn

from .unet_funcs import DownSampling, OutConv, DoubleConv, SingleConv


class UNet3Plus(nn.Module):
    """ UNet3++ with deep supervision """

    def __init__(self, n_channels, n_classes):
        super(UNet3Plus, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes

        self.num_channels = [15, 30, 60, 120, 240]

        # ------------------------ Encoder ------------------------
        self.input_layer = DoubleConv(self.n_channels, self.num_channels[0])
        self.down1 = DownSampling(self.num_channels[0], self.num_channels[1])
        self.down2 = DownSampling(self.num_channels[1], self.num_channels[2])
        self.down3 = DownSampling(self.num_channels[2], self.num_channels[3])
        self.down4 = DownSampling(self.num_channels[3], self.num_channels[4])

        # ------------------------ Decoder ------------------------
        self.CatChannels = 32
        self.CatBlocks = 5
        self.UpChannels = self.CatChannels * self.CatBlocks

        """ stage 4d """
        # h1 -> 512 * 512, hd4 -> 64 * 64, Pooling using 8 * 8 kernels
        self.h1_PT_hd4_pool = nn.MaxPool2d(kernel_size=(8, 8), stride=(8, 8))
        self.h1_PT_hd4_conv = nn.Conv2d(self.num_channels[0], self.CatChannels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.h1_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd4_relu = nn.ReLU(inplace=True)
        self.h1_PT_hd4 = nn.Sequential(self.h1_PT_hd4_pool, self.h1_PT_hd4_conv, self.h1_PT_hd4_bn, self.h1_PT_hd4_relu)

        # h2 -> 256 * 256, hd4 -> 64 * 64, Pooling using 4 * 4 kernels
        self.h2_PT_hd4_pool = nn.MaxPool2d(kernel_size=(4, 4), stride=(4, 4))
        self.h2_PT_hd4_conv = nn.Conv2d(self.num_channels[1], self.CatChannels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.h2_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd4_relu = nn.ReLU(inplace=True)
        self.h2_PT_hd4 = nn.Sequential(self.h2_PT_hd4_pool, self.h2_PT_hd4_conv, self.h2_PT_hd4_bn, self.h2_PT_hd4_relu)

        # h3 -> 128 * 128, hd4 -> 64 * 64, Pooling using 2 * 2 kernels
        self.h3_PT_hd4_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.h3_PT_hd4_conv = nn.Conv2d(self.num_channels[2], self.CatChannels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.h3_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_PT_hd4_relu = nn.ReLU(inplace=True)
        self.h3_PT_hd4 = nn.Sequential(self.h3_PT_hd4_pool, self.h3_PT_hd4_conv, self.h3_PT_hd4_bn, self.h3_PT_hd4_relu)

        # h4 -> 64 * 64, hd4 -> 64 * 64, no pooling
        self.h4_PT_hd4_conv = nn.Conv2d(self.num_channels[3], self.CatChannels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.h4_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h4_PT_hd4_relu = nn.ReLU(inplace=True)
        self.h4_PT_hd4 = nn.Sequential(self.h4_PT_hd4_conv, self.h4_PT_hd4_bn, self.h4_PT_hd4_relu)

        # h5 -> 32 * 32, hd4 -> 64 * 64, upsampling
        self.h5_UT_hd4_up = nn.Upsample(scale_factor=2, mode="bilinear")
        self.h5_UT_hd4_conv = nn.Conv2d(self.num_channels[4], self.CatChannels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.h5_UT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h5_UT_hd4_relu = nn.ReLU(inplace=True)
        self.h5_UT_hd4 = nn.Sequential(self.h5_UT_hd4_up, self.h5_UT_hd4_conv, self.h5_UT_hd4_bn, self.h5_UT_hd4_relu)

        # fusion features from h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_PT_hd4, and h5_UT_hd4
        self.double_conv_hd4 = DoubleConv(self.UpChannels, self.UpChannels)

        """ stage 3d """
        # h1 -> 512 * 512, hd3 -> 128 * 128, Pooling using 4 * 4 kernels
        self.h1_PT_hd3_pool = nn.MaxPool2d(kernel_size=(4, 4), stride=(4, 4))
        self.h1_PT_hd3_conv = nn.Conv2d(self.num_channels[0], self.CatChannels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.h1_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd3_relu = nn.ReLU(inplace=True)
        self.h1_PT_hd3 = nn.Sequential(self.h1_PT_hd3_pool, self.h1_PT_hd3_conv, self.h1_PT_hd3_bn, self.h1_PT_hd3_relu)

        # h2 -> 256 * 256, hd3 -> 128 * 128, Pooling using 2 * 2 kernels
        self.h2_PT_hd3_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.h2_PT_hd3_conv = nn.Conv2d(self.num_channels[1], self.CatChannels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.h2_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd3_relu = nn.ReLU(inplace=True)
        self.h2_PT_hd3 = nn.Sequential(self.h2_PT_hd3_pool, self.h2_PT_hd3_conv, self.h2_PT_hd3_bn, self.h2_PT_hd3_relu)

        # h3 -> 128 * 128, hd3 -> 128 * 128, no pooling
        self.h3_PT_hd3_conv = nn.Conv2d(self.num_channels[2], self.CatChannels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.h3_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_PT_hd3_relu = nn.ReLU(inplace=True)
        self.h3_PT_hd3 = nn.Sequential(self.h3_PT_hd3_conv, self.h3_PT_hd3_bn, self.h3_PT_hd3_relu)

        # h4 -> 64 * 64, hd3 -> 128 * 128, upsampling
        self.hd4_UT_hd3_up = nn.Upsample(scale_factor=2, mode="bilinear")
        self.hd4_UT_hd3_conv = nn.Conv2d(self.UpChannels, self.CatChannels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.hd4_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd3_relu = nn.ReLU(inplace=True)
        self.hd4_UT_hd3 = nn.Sequential(self.hd4_UT_hd3_up, self.hd4_UT_hd3_conv, self.hd4_UT_hd3_bn, self.hd4_UT_hd3_relu)

        # h5 -> 32 * 32, hd4 -> 64 * 64, upsampling
        self.h5_UT_hd3_up = nn.Upsample(scale_factor=4, mode="bilinear")
        self.h5_UT_hd3_conv = nn.Conv2d(self.num_channels[4], self.CatChannels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.h5_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h5_UT_hd3_relu = nn.ReLU(inplace=True)
        self.h5_UT_hd3 = nn.Sequential(self.h5_UT_hd3_up, self.h5_UT_hd3_conv, self.h5_UT_hd3_bn, self.h5_UT_hd3_relu)

        # fusion features from h1_PT_hd3, h2_PT_hd3, h3_PT_hd3, hd4_UT_hd3, and h5_UT_hd3
        self.double_conv_hd3 = DoubleConv(self.UpChannels, self.UpChannels)

        """ stage 2d """
        # h1 -> 512 * 512, hd2 -> 256 * 256, Pooling using 2 * 2 kernels
        self.h1_PT_hd2_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.h1_PT_hd2_conv = nn.Conv2d(self.num_channels[0], self.CatChannels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.h1_PT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd2_relu = nn.ReLU(inplace=True)
        self.h1_PT_hd2 = nn.Sequential(self.h1_PT_hd2_pool, self.h1_PT_hd2_conv, self.h1_PT_hd2_bn, self.h1_PT_hd2_relu)

        # h2 -> 256 * 256, hd2 -> 256 * 256, no pooling
        self.h2_PT_hd2_conv = nn.Conv2d(self.num_channels[1], self.CatChannels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.h2_PT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd2_relu = nn.ReLU(inplace=True)
        self.h2_PT_hd2 = nn.Sequential(self.h2_PT_hd2_conv, self.h2_PT_hd2_bn, self.h2_PT_hd2_relu)

        # hd3 -> 128 * 128, hd2 -> 256 * 256, upsampling
        self.hd3_UT_hd2_up = nn.Upsample(scale_factor=2, mode="bilinear")
        self.hd3_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.hd3_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd2_relu = nn.ReLU(inplace=True)
        self.hd3_UT_hd2 = nn.Sequential(self.hd3_UT_hd2_up, self.hd3_UT_hd2_conv, self.hd3_UT_hd2_bn, self.hd3_UT_hd2_relu)

        # hd4 -> 64 * 64, hd2 -> 256 * 256, upsampling
        self.hd4_UT_hd2_up = nn.Upsample(scale_factor=4, mode="bilinear")
        self.hd4_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.hd4_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd2_relu = nn.ReLU(inplace=True)
        self.hd4_UT_hd2 = nn.Sequential(self.hd4_UT_hd2_up, self.hd4_UT_hd2_conv, self.hd4_UT_hd2_bn, self.hd4_UT_hd2_relu)

        # h5 -> 32 * 32, hd2 -> 256 * 256, upsampling
        self.h5_UT_hd2_up = nn.Upsample(scale_factor=8, mode="bilinear")
        self.h5_UT_hd2_conv = nn.Conv2d(self.num_channels[4], self.CatChannels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.h5_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h5_UT_hd2_relu = nn.ReLU(inplace=True)
        self.h5_UT_hd2 = nn.Sequential(self.h5_UT_hd2_up, self.h5_UT_hd2_conv, self.h5_UT_hd2_bn, self.h5_UT_hd2_relu)

        # fusion features from h1_PT_hd3, h2_PT_hd3, h3_PT_hd3, hd4_UT_hd3, and h5_UT_hd3
        self.double_conv_hd2 = DoubleConv(self.UpChannels, self.UpChannels)

        """ stage 1d """
        # h1 -> 512 * 512, hd1 -> 512 * 512, no pooling
        self.h1_PT_hd1_conv = nn.Conv2d(self.num_channels[0], self.CatChannels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.h1_PT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd1_relu = nn.ReLU(inplace=True)
        self.h1_PT_hd1 = nn.Sequential(self.h1_PT_hd1_conv, self.h1_PT_hd1_bn, self.h1_PT_hd1_relu)

        # hd2 -> 256 * 256, hd1 -> 512 * 512, upsampling
        self.hd2_UT_hd1_up = nn.Upsample(scale_factor=2, mode="bilinear")
        self.hd2_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.hd2_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd2_UT_hd1_relu = nn.ReLU(inplace=True)
        self.hd2_UT_hd1 = nn.Sequential(self.hd2_UT_hd1_up, self.hd2_UT_hd1_conv, self.hd2_UT_hd1_bn, self.hd2_UT_hd1_relu)

        # hd3 -> 128 * 128, hd1 -> 512 * 512, upsampling
        self.hd3_UT_hd1_up = nn.Upsample(scale_factor=4, mode="bilinear")
        self.hd3_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.hd3_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd1_relu = nn.ReLU(inplace=True)
        self.hd3_UT_hd1 = nn.Sequential(self.hd3_UT_hd1_up, self.hd3_UT_hd1_conv, self.hd3_UT_hd1_bn, self.hd3_UT_hd1_relu)

        # hd4 -> 64 * 64, hd1 -> 512 * 512, upsampling
        self.hd4_UT_hd1_up = nn.Upsample(scale_factor=8, mode="bilinear")
        self.hd4_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.hd4_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd1_relu = nn.ReLU(inplace=True)
        self.hd4_UT_hd1 = nn.Sequential(self.hd4_UT_hd1_up, self.hd4_UT_hd1_conv, self.hd4_UT_hd1_bn, self.hd4_UT_hd1_relu)

        # h5 -> 32 * 32, hd1 -> 512 * 512, upsampling
        self.h5_UT_hd1_up = nn.Upsample(scale_factor=16, mode="bilinear")
        self.h5_UT_hd1_conv = nn.Conv2d(self.num_channels[4], self.CatChannels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.h5_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.h5_UT_hd1_relu = nn.ReLU(inplace=True)
        self.h5_UT_hd1 = nn.Sequential(self.h5_UT_hd1_up, self.h5_UT_hd1_conv, self.h5_UT_hd1_bn, self.h5_UT_hd1_relu)

        # fusion features from h1_PT_hd3, h2_PT_hd3, h3_PT_hd3, hd4_UT_hd3, and h5_UT_hd3
        self.double_conv_hd1 = DoubleConv(self.UpChannels, self.UpChannels)

        self.output_layer = OutConv(self.UpChannels, self.n_classes)

    def forward(self, x):
        # ------------------------ Encoder ------------------------
        h1 = self.input_layer(x)            # h1 -> 512 * 512
        h2 = self.down1(h1)                 # h2 -> 256 * 256
        h3 = self.down2(h2)                 # h3 -> 128 * 128
        h4 = self.down3(h3)                 # h4 -> 64 * 64
        h5 = self.down4(h4)                 # h5 -> 32 * 32

        # ------------------------ Decoder ------------------------
        h1_hd4 = self.h1_PT_hd4(h1)
        h2_hd4 = self.h2_PT_hd4(h2)
        h3_hd4 = self.h3_PT_hd4(h3)
        h4_hd4 = self.h4_PT_hd4(h4)
        h5_hd4 = self.h5_UT_hd4(h5)
        hd4 = self.double_conv_hd4(torch.cat((h1_hd4, h2_hd4, h3_hd4, h4_hd4, h5_hd4), 1))  # hd4 -> 64 * 64

        h1_hd3 = self.h1_PT_hd3(h1)
        h2_hd3 = self.h2_PT_hd3(h2)
        h3_hd3 = self.h3_PT_hd3(h3)
        hd4_hd3 = self.hd4_UT_hd3(hd4)
        h5_hd3 = self.h5_UT_hd3(h5)
        hd3 = self.double_conv_hd3(torch.cat((h1_hd3, h2_hd3, h3_hd3, hd4_hd3, h5_hd3), 1))    # hd3 -> 128 * 128

        h1_hd2 = self.h1_PT_hd2(h1)
        h2_hd2 = self.h2_PT_hd2(h2)
        hd3_hd2 = self.hd3_UT_hd2(hd3)
        hd4_hd2 = self.hd4_UT_hd2(hd4)
        h5_hd2 = self.h5_UT_hd2(h5)
        hd2 = self.double_conv_hd2(torch.cat((h1_hd2, h2_hd2, hd3_hd2, hd4_hd2, h5_hd2), 1))    # hd2 -> 256 * 256

        h1_hd1 = self.h1_PT_hd1(h1)
        hd2_hd1 = self.hd2_UT_hd1(hd2)
        hd3_hd1 = self.hd3_UT_hd1(hd3)
        hd4_hd1 = self.hd4_UT_hd1(hd4)
        h5_hd1 = self.h5_UT_hd1(h5)
        hd1 = self.double_conv_hd1(torch.cat((h1_hd1, hd2_hd1, hd3_hd1, hd4_hd1, h5_hd1), 1))   # hd1 -> 512 * 512

        output = self.output_layer(hd1)

        return output
