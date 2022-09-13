#! python3
# encoding: utf-8
"""
@author:    Yize Wang
@contact:    wangyize@hust.edu.cn
@file:    eval.py
@time:   2022/3/16 18:26
@description:    
"""

import torch
from tqdm import tqdm
from loss.bceloss import BCE_loss
from loss.diceloss import DiceLoss


def eval_net(net, loader, device, n_val):
    """ n_val is total number of validation images, but not validation batches """

    net.eval()              # call this function when evaluating the network

    tot = 0
    with torch.no_grad():
        with tqdm(total=n_val, desc="Validation round", unit="image", leave=False) as pbar:
            for batch in loader:
                images = batch["image"]
                true_masks = batch["mask"]

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32)

                # predict using the network
                mask_pred = net(images)

                dice_loss = DiceLoss(mask_pred, true_masks)
                tot += dice_loss().cpu().item()

                pbar.update(images.size()[0])

    return tot / len(loader)
