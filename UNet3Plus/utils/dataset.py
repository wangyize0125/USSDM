#! python3
# encoding: utf-8
"""
@author:    Yize Wang
@contact:    wangyize@hust.edu.cn
@file:    dataset.py
@time:   2022/3/16 16:46
@description:    
"""

import os
import logging
import torch
import numpy as np

from os.path import splitext
from os import listdir
from glob import glob
from torch.utils.data import Dataset
from PIL import Image


"""
Dataset settings:
    image in one folder
    masks in one folder
    mask file should have same prefix as that of the original image
        eg: 0001 ----> 0001*
"""


class BasicDataset(Dataset):
    def __init__(self, unet_type, imgs_dir, masks_dir, scale=0.5):
        self.unet_type = unet_type
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        assert 0 < self.scale <= 1, "Image scale must be between 0 and 1"

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir) if not file.startswith(".")]
        logging.info("Creating dataset with {} examples".format(len(self.ids)))

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, unet_type, pil_img, scale, mask=True):
        # resize the image first
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, "Scale is too small"
        pil_img = pil_img.resize((newW, newH))

        # binary
        if mask:
            pil_img = pil_img.convert("L")      # after binary, image only one channel
        else:
            pass

        # add a dimension if there is only 2 dims
        img_nd = np.array(pil_img)
        if len(img_nd.shape) == 2:
            logging.info("Image only have 2 dimensions when preprocess in BasicDataset")
            img_nd = np.expand_dims(img_nd, axis=2)

        # HxWxC to CxHxW
        img_trans = img_nd.transpose((2, 0, 1))

        if img_trans.max() > 1 and mask:         # convert non-zero to 1
            img_trans = img_trans / img_trans.max()     # mask convert to 0 and 1
        elif img_trans.max() > 1 and not mask:
            img_trans = img_trans / 255                 # original image not convert to 0 and 1
        else:
            pass

        return img_trans

    def __getitem__(self, item):
        idx = self.ids[item]

        mask_file = glob(os.path.join(self.masks_dir, idx + "*"))
        img_file = glob(os.path.join(self.imgs_dir, idx + "*"))

        assert len(mask_file) == 1, "Either no mask or multiple masks found for the ID {}: {}".format(idx, mask_file)
        assert len(img_file) == 1, "Either no image or multiple images found for the ID {}: {}".format(idx, img_file)
        mask = Image.open(mask_file[0])
        image = Image.open(img_file[0])

        assert image.size == mask.size, "Image and mask {} should have the same size, but are not the same".format(idx)
        image = self.preprocess(self.unet_type, image, self.scale, mask=False)
        mask = self.preprocess(self.unet_type, mask, self.scale)

        return {"image": torch.from_numpy(image), "mask": torch.from_numpy(mask)}


if __name__ == "__main__":
    dataset = BasicDataset("v1", "../data/train/imgs/", "../data/train/masks/", 1.0)
    img1 = dataset[0]
    print("image size: {}".format(img1["image"].size()))
    print("mask size: {}".format(img1["mask"].size()))
    print("image maximum: {}".format(torch.max(img1["image"])))
    print("mask maximum: {}".format(torch.max(img1["mask"])))
