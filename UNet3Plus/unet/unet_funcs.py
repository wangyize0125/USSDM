#! python3
# encoding: utf-8
"""
@author:    Yize Wang
@contact:    wangyize@hust.edu.cn
@file:    unet_funcs.py
@time:   2022/3/16 15:35
@description:    define unet backbone layers
"""

import torch
import logging
import torch.nn as nn
import torch.nn.functional as func


"""
Remark about Conv2d:
    Input for Conv2d should be NxCxHxW, where N is batch size, C is pixel value, N is demanded
    Configurations in these codes ensure a same dimension of input and output
"""


class DoubleConv(nn.Module):
    """ (convolution 3 * 3 -> [Batch normalization] -> ReLU) * [2 times] """

    def __init__(self, in_channels, out_channels):
        """ in and out channels are input and output dimension, but not the data """

        super(DoubleConv, self).__init__()

        # define the sequence
        self.double_conv = nn.Sequential()
        self.double_conv.add_module("Conv1_3x3", nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1))
        self.double_conv.add_module("Batch_norm1", nn.BatchNorm2d(out_channels))
        self.double_conv.add_module("ReLU1", nn.ReLU(inplace=True))
        self.double_conv.add_module("Conv2_3x3", nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1))
        self.double_conv.add_module("Batch_norm2", nn.BatchNorm2d(out_channels))
        self.double_conv.add_module("ReLU2", nn.ReLU(inplace=True))

    def forward(self, x):
        return self.double_conv(x)


class SingleConv(nn.Module):
    """ (convolution 3 * 3 -> [Batch normalization] -> ReLU) """

    def __init__(self, in_channels, out_channels):
        """ in and out channels are input and output dimension, but not the data """

        super(SingleConv, self).__init__()

        # define the sequence
        self.double_conv = nn.Sequential()
        self.double_conv.add_module("Conv1_3x3", nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1))
        self.double_conv.add_module("Batch_norm1", nn.BatchNorm2d(out_channels))
        self.double_conv.add_module("ReLU1", nn.ReLU(inplace=True))

    def forward(self, x):
        return self.double_conv(x)


class DownSampling(nn.Module):
    """ Down sampling using maxpool_2x2 then following double conv """

    def __init__(self, in_channels, out_channels):
        super(DownSampling, self).__init__()

        self.down_sampling = nn.Sequential()
        self.down_sampling.add_module("Max_pool", nn.MaxPool2d(kernel_size=(2, 2)))
        self.down_sampling.add_module("DoubleConv", DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.down_sampling(x)


class UpSampling(nn.Module):
    """ Up sampling using decomposed convolution then following a double conv """

    def __init__(self, in_channels, out_channels):
        super(UpSampling, self).__init__()

        assert in_channels % 2 == 0, "In channels in Up_sampling should be even, but [{}] accepted".format(in_channels)

        self.up_conv = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=(2, 2), stride=(2, 2))
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """ x1 is up sampled feature and x2 is down sampled feature """

        x1 = self.up_conv(x1)            # up sampling x1 first

        # concatenates two features
        # calculate dimension differences between two inputs
        # generally, they should have a same dimension
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]]).item()
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]]).item()
        if diffY != 0 or diffX != 0:
            logging.warning("x1 and x2 have different dimensions when fusing features!")

            x1 = func.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])     # padding

        x = torch.cat([x2, x1], dim=1)

        return self.double_conv(x)


class OutConv(nn.Module):
    """ output layer """

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()

        self.out_layer = nn.Sequential()
        self.out_layer.add_module("Conv", nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1)))
        self.out_layer.add_module("Activate", nn.Sigmoid())

    def forward(self, x):
        return self.out_layer(x)
