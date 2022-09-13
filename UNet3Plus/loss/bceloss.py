#! python3
# encoding: utf-8
"""
@author:    Yize Wang
@contact:    wangyize@hust.edu.cn
@file:    bceloss.py
@time:   2022/3/16 18:27
@description:    
"""

import torch
import torch.nn as nn


def BCE_loss(pred, label):
    """ used for binary classification """
    bce_loss = nn.BCELoss(size_average=True)
    bce_out = bce_loss(pred, label)

    return bce_out
