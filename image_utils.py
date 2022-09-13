#!python3
# -*- coding:utf-8 -*-
# @author: Yize Wang
# @email: wangyize0125@hotmail.com
# @time: 2022/4/5 16:14

import cv2 as cv
import numpy as np
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt

from output_utils import print_format, print_warning


def is_binary(img: np.ndarray) -> bool:
    """
    check the image is binary file or not
    :param img: image array
    :return: bool, is or not is
    """

    flag = True
    if len(img.shape) > 2:
        flag = False
    else:
        if np.argwhere(0 < flag < 255).size > 0:
            flag = False

    return flag


def to_grey(source_file: str, output_file: str):
    """
    convert source file into grey style
    :param source_file: filename
    :param output_file: filename
    :return: None
    """

    source = np.array(Image.open(source_file))
    assert len(source.shape) == 3, "Source image should have three channels when converting to grey format"

    source = Image.fromarray(source).convert("L")

    source.save(output_file)


def edge_detection(source_file: str, output_file: str, threshold_min: int, threshold_max: int):
    """
    detect edge in images
    :param source_file: filename
    :param output_file: filename
    :param threshold_min: min threshold where edges smaller than this threshold are non-edges
    :param threshold_max: max threshold where edges larger than this threshold are edges
    :return: None
    """

    img = cv.imread(source_file)
    edges = cv.Canny(img, threshold_min, threshold_max)

    cv.imwrite(output_file, edges)


def corner_detection(source_file: str, output_file: str, max_corners: int, quality_level: float, min_distance: float, refine: bool):
    """
    detect corners in the image
    :param source_file: filename, grey style file is required
    :param output_file: filename
    :param max_corners: maximum corners to find
    :param quality_level: percent, which will multiply maximum quality be the threshold
    :param min_distance: minimum distance between two found corners
    :param refine: refine sub pixel accuracy flag
    :return: corners
    """

    img = cv.imread(source_file)
    if len(img.shape) > 2:
        error = np.sum((img[:, :, 0] - img[:, :, 1]) + (img[:, :, 0] - img[:, :, 2]) + (img[:, :, 1] - img[:, :, 2]))
        assert error == 0, "Grey image with one channel is accepted in corner detection!"

    corners = cv.goodFeaturesToTrack(img[:, :, 0], max_corners, quality_level, min_distance)     # find corner

    # refine sub pixel accuracy
    if refine:
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        corners = cv.cornerSubPix(img[:, :, 0], corners, (5, 5), (-1, -1), criteria)

    # plot image
    fig = plt.figure(figsize=(8, 8 / img.shape[0] * img.shape[1]))
    plt.axis("off")
    plt.imshow(img)
    corners = corners.reshape((-1, 2))
    plt.scatter(corners[:, 0], corners[:, 1], s=2, c="blue", marker="o")
    plt.savefig(output_file)


def to_binary(source_file: str, output_file: str, threshold: int, filter_flag: bool, factor: float):
    """
    convert image into binary image, and store it in output_file
    :param source_file: filename
    :param output_file: filename
    :param threshold: smaller than is 0, greater than is 255 (threshold < 0, calculate it automatically)
    :param filter_flag: filter for discard noise
    :param factor: factor to calculate threshold
    :return: None
    """

    source = np.array(Image.open(source_file))
    if len(source.shape) == 3:
        print_format("Input channel is 3, convert to grey style")
        to_grey(source_file, output_file)       # grey first
        source = np.array(Image.open(output_file))
    else:
        pass

    if filter_flag:
        source = np.array(Image.fromarray(source).filter(ImageFilter.GaussianBlur))

    # calculate threshold
    if threshold < 0:
        # calculate mean boundary
        mean_boundary = 0
        max_boundary = 0
        total_num = 0
        for i in range(source.shape[0]):
            idx = np.argwhere(source[i] < 255)
            if idx.size > 0:
                total_num += 1
                mean_boundary += (source[i, idx[0]] + source[i, idx[-1]]) / 2
                max_boundary = max(source[i, idx[0]] + source[i, idx[-1]])
        mean_boundary /= total_num
        mean_boundary = np.min(source)

        # calculate mean all and mean boundary
        mean_all = np.mean(source[np.where(source < 255)])

        # new threshold
        threshold = mean_boundary + (mean_all - mean_boundary) * factor

    source[np.where(source >= threshold)] = 255
    source[np.where(source < threshold)] = 0

    output = Image.fromarray(source)
    output.save(output_file)


def crop_image(img_file: str, anchor: tuple or list, size: tuple or list, output: str) -> tuple:
    """
    crop image file using anchor and size
    :param img_file: image file name
    :param anchor: start on left up corner
    :param size: crop size
    :param output: store name
    :return: None
    """

    img = np.array(Image.open(img_file))

    # convert them to int
    anchor, size = [int(item) for item in anchor], [int(item) for item in size]
    # col is row, row is col
    anchor, size = [anchor[1], anchor[0]], [size[1], size[0]]

    # modify anchor and size if necessary
    assert size[0] <= img.shape[0] and size[1] <= img.shape[1], "Size is greater than image size!"
    if anchor[0] < 0 or anchor[0] > img.shape[0] - 1:
        print_warning("Anchor is out of image, which will be set to 0!")
        anchor[0] = 0
    if anchor[1] < 0 or anchor[1] > img.shape[1] - 1:
        print_warning("Anchor is out of image, which will be set to 0!")
        anchor[1] = 0
    for i in range(2):
        if anchor[i] + size[i] > img.shape[i]:
            print_warning("Size is out of image, anchor will be shifted!")
            anchor[i] += (img.shape[i] - anchor[i] - size[i])

    img = Image.fromarray(img[anchor[0]: anchor[0] + size[0], anchor[1]: anchor[1] + size[1]])
    img.save(output)

    return anchor[1], anchor[0]


if __name__ == "__main__":
    # to_grey("../../fuse.png", "../../grey.png")
    # edge_detection("../../grey.png", "../../binary.png", 100, 200)
    # to_binary("../../grey.png", "../../binary.png", -1, False)
    # corner_detection("../../grey.png", "../../corners.png", 8, 0.5, 0.1, True)
    # print(is_binary(np.array(Image.open("../../binary.png"))))
    image = crop_image("../../calculation/MVI_7188/54.0206_un_distort.png", (640, 0), (1056, 1056))
    image.save("../../test_crop.png")
