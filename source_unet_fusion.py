#!python3
# -*- coding:utf-8 -*-
# @author: Yize Wang
# @email: wangyize0125@hotmail.com
# @time: 2022/4/5 15:49

import numpy as np
from PIL import Image


def fuse_source_unet(source_file: str, unet_file: str, output_file: str):
    """
    fuse source and unet3+ predicted images, and store in output file
    :param source_file: filename
    :param unet_file: filename
    :param output_file: output filename
    """

    source, unet = np.array(Image.open(source_file)), np.array(Image.open(unet_file))
    assert len(unet.shape) == 2, "UNet predicted image can only has one channel!"

    source[np.where(unet == 0)] = np.ones(3) * 255      # fuse together
    output = Image.fromarray(source)
    output.save(output_file)


if __name__ == "__main__":
    fuse_source_unet("../../data/train/test/resized_004.png", "../../data/train/test_predict/resized_004.png", "../../fuse.png")
