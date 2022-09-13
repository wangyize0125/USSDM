#!python3
# -*- coding:utf-8 -*-
# @author: Yize Wang
# @email: wangyize0125@hotmail.com
# @time: 2022/4/9 20:36

import os
import argparse

import cv2
import torch
import logging
import numpy as np
from scipy.optimize import leastsq

from .unet import UNet, UNetV3
from PIL import Image
from .utils.dataset import BasicDataset
from torchvision import transforms


def get_args():
    parser = argparse.ArgumentParser(description="Predict from input images")

    parser.add_argument("-u", "--unet_type", metavar="U", default="v1", help="UNet type v1/v2/v3")
    parser.add_argument("-m", "--model", default="MODEL.pth", metavar="FILE", help="Specify the model file")
    parser.add_argument("-i", "--input", metavar="INPUT", help="filenames of input images", required=True)
    parser.add_argument("-o", "--output", metavar="OUTPUT", help="filenames of output images")
    parser.add_argument("-n", "--no_save", action="store_true", help="Do not save the output", default=False)
    parser.add_argument("-t", "--threshold", type=float, help="Minimum probability value to consider a mask white", default=0.5)
    parser.add_argument("-s", "--scale", type=float, help="image scale factor when preprocessing", default=1.0)

    return parser.parse_args()


def predict_img(unet_type, net, full_img, scale_factor=1, out_threshold=0.5):
    net.eval()

    img = torch.from_numpy(BasicDataset.preprocess(unet_type, full_img, scale_factor, mask=False).astype(np.float32))
    img = img.unsqueeze(0)

    with torch.no_grad():
        output = net(img)
        output = output.squeeze(0)

        tf = transforms.Compose([transforms.ToPILImage(), transforms.Resize(full_img.size[1]), transforms.ToTensor()])
        probs = tf(output)
        full_mask = probs.squeeze().numpy()

    return (full_mask > out_threshold).astype(np.float32).astype(np.uint8)


def predict_and_save(unet_type, net, full_img, scale_factor=1, out_threshold=0.5, output_filename="", post_pro=True, resize_size=(0, 0)):
    net.eval()

    img = torch.from_numpy(BasicDataset.preprocess(unet_type, full_img, scale_factor, mask=False).astype(np.float32))
    img = img.unsqueeze(0)

    with torch.no_grad():
        output = net(img)
        output = output.squeeze(0)

        tf = transforms.Compose([transforms.ToPILImage(), transforms.Resize(full_img.size[1]), transforms.ToTensor()])
        probs = tf(output)
        full_mask = probs.squeeze().numpy()

    mask_img = (full_mask > out_threshold).astype(np.float32).astype(np.uint8)

    if post_pro:
        # fill 1 first
        for i in range(mask_img.shape[0]):
            idxes = np.argwhere(mask_img[i] == 1).flatten()

            if idxes.size > 0:
                mask_img[i, idxes[0]: idxes[-1]] = 1

        # use a circle to represent the target directly
        contours, hierachy = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_r = 0
        center = None
        for i in range(len(contours)):
            if contours[i].shape[0] > 5:
                rrt = cv2.fitEllipse(contours[i])

                if np.sum(rrt[1]) > max_r and rrt[1][0] < 512 and rrt[1][1] < 512:
                    max_r = np.sum(rrt[1])
                    center = list(rrt[0])
        if abs(center[0] - 256) > 100 or abs(center[1] - 256) > 100:
            center[0] = center[1] = 256

        # max radius
        max_r = 0
        for i in range(mask_img.shape[0]):
            idxes = np.argwhere(mask_img[i] == 1).flatten()

            if idxes.size > 0:
                if ((idxes[0] - center[0]) ** 2 + (i - center[1]) ** 2) ** 0.5 > max_r:
                    max_r = ((idxes[0] - center[0]) ** 2 + (i - center[1]) ** 2) ** 0.5
                if ((idxes[-1] - center[0]) ** 2 + (i - center[1]) ** 2) ** 0.5 > max_r:
                    max_r = ((idxes[-1] - center[0]) ** 2 + (i - center[1]) ** 2) ** 0.5
        max_r = int(max_r) + 1

        # correct predict result
        for i in range(mask_img.shape[0]):
            if abs(i - center[1]) < max_r:
                left_idx = int(center[0] - (max_r ** 2 - (i - center[1]) ** 2) ** 0.5)
                right_idx = int(center[0] + (max_r ** 2 - (i - center[1]) ** 2) ** 0.5)

                left_idx = 0 if left_idx < 0 else left_idx
                right_idx = 511 if right_idx > 511 else right_idx

                mask_img[i, left_idx: right_idx + 1] = 1

    mask_img = np.array(mask_to_image(mask_img).resize(resize_size))
    mask_img[np.where(mask_img != 0)] = 255

    result_img = Image.fromarray(mask_img)
    result_img.save(output_filename[:-4] + ".png")


def mask_to_image(mask):
    return Image.fromarray((mask * 255))


if __name__ == "__main__":
    args = get_args()

    unet_type = args.unet_type
    in_file_folder = args.input
    out_file_folder = args.output

    if unet_type == "v2":
        pass
    elif unet_type =="v3":
        net = UNetV3(n_channels=3, n_classes=1)
    else:
        net = UNet(n_channels=3, n_classes=1)

    logging.info("Loading model {}".format(args.model))

    model = torch.load(args.model, map_location=torch.device("cpu"))
    net.load_state_dict(model)
    logging.info("Model loaded!")

    for root, dirs, files in os.walk(args.input):
        for file in files:
            if file.endswith(".png"):
                logging.info("\nPredicting image {} ...".format(file))
                img = Image.open(os.path.join(args.input, file))
                mask = predict_img(unet_type=unet_type, net=net, full_img=img, scale_factor=args.scale, out_threshold=args.threshold)

                if not args.no_save:
                    result = mask_to_image(mask)
                    result.save(os.path.join(args.output, file))
                    logging.info("Mask saved to {}".format(os.path.join(args.output, file)))
