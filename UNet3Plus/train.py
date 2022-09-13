#! python3
# encoding: utf-8
"""
@author:    Yize Wang
@contact:    wangyize@hust.edu.cn
@file:    train.py
@time:   2022/3/16 16:40
@description:    
"""

import os
import sys
import math
import torch
import logging
import argparse
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler as lr_scheduler

from apex import amp
from unet import UNet, UNetV3
from tqdm import tqdm
from torch import optim
from loss.diceloss import DiceLoss
from utils.eval import eval_net
from utils.dataset import BasicDataset
from torch.utils.data import random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

dir_img = "./data/train/imgs/"
dir_mask = "./data/train/masks/"
dir_checkpoint = "./ckpts/"


def train_net(unet_type, net, device, epochs=100, batch_size=10, lr=0.01, val_percent=0.2, save_cp=True, img_scale=0.5):
    # create dataset inclusing train and validation dataset
    dataset = BasicDataset(unet_type, dir_img, dir_mask, img_scale)
    n_val = int(len(dataset) * val_percent)     # number of validation img
    n_train = len(dataset) - n_val              # number of train img
    assert n_val > 0 and n_train > 0, "Train or validation set is empty"

    # train and vilidation set
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    writer = SummaryWriter(comment="LR_{}_BS_{}_SCALE_{}".format(lr, batch_size, img_scale))

    global_step = 0

    logging.info(
        """Start training:
                UNet type:          {}
                Epochs:             {}
                Batch size:         {}
                Learning rate:      {}
                Dataset size:       {}
                Training size:      {}
                Validation size:    {}
                Checkpoints:        {}
                Device:             {}
                Images scaling:     {}
        """.format(unet_type, epochs, batch_size, lr, len(dataset), n_train, n_val, save_cp, device.type, img_scale)
    )

    # optimizer accelerated using amp
    optimizer = optim.RMSprop([{"params": net.parameters(), "initial_lr": lr}], lr=lr, weight_decay=1e-8)
    model, optimizer = amp.initialize(net, optimizer, opt_level="O0")

    # automatically tune learning rate
    lf = lambda x: (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * 0.95 + 0.05
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf, last_epoch=global_step)

    lrs, train_loss, val_loss = [], [], []
    best_loss = 100000
    for epoch in range(epochs):
        cul_lr = optimizer.param_groups[0]["lr"]
        print("\nEpoch = {}, lr = {}".format(epoch + 1, cul_lr))

        net.train()         # set train flag of the network

        epoch_loss = 0

        with tqdm(total=n_train, desc="Epoch {} / {}".format(epoch + 1, epochs), unit="image") as pbar:
            for batch in train_loader:
                imgs = batch["image"]
                true_masks = batch["mask"]
                assert imgs.shape[1] == net.n_channels, "Network has been defined with {} input channels, " \
                    "but loaded images have {} channels".format(net.n_channels, imgs.shape[1])
                # move to GPU
                imgs = imgs.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32)

                optimizer.zero_grad()           # clear gradient

                masks_pred = net(imgs)

                # loss function
                criterion = DiceLoss(masks_pred, true_masks)
                loss = criterion()
                epoch_loss += loss.item()

                writer.add_scalar("Loss/train", loss.item(), global_step + 1)
                pbar.set_postfix(**{"loss (batch)": loss.item()})

                # update hyper-parameters
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1

        epoch_loss /= len(train_loader)

        # evaluate validation set predict accuracy after each epoch
        val_score = eval_net(net, val_loader, device, n_val)
        logging.info("Validation Dice coeff: {}".format(val_score))
        writer.add_scalar("Dice/test", val_score, global_step)

        val_loss.append(val_score)

        scheduler.step()
        lrs.append(cul_lr)
        train_loss.append(epoch_loss)

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info("Created checkpoint directory")
            except OSError:
                pass

            if epoch_loss < best_loss:
                torch.save(net.state_dict(), dir_checkpoint + "CP_epoch{}_loss_{}.pth".format(epoch + 1, str(epoch_loss)))
                best_loss = epoch_loss
                logging.info("Checkpoint {} saved ! loss (batch) = {}".format(epoch + 1, str(epoch_loss)))

    fig = plt.figure()
    ax = plt.subplot(121)
    plt.plot(lrs, ".-", label="LambdaLR")
    plt.xlabel("epoch")
    plt.ylabel("LR")
    ax = plt.subplot(122)
    plt.plot(train_loss, "r.-", label="Train loss")
    plt.plot(val_loss, "b:", label="Val loss")
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig("output.png", dpi=300)

    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description="Train the UNet on images and target masks")

    parser.add_argument("-g", "--gpu_id", dest="gpu_id", metavar="G", type=int, default=0, help="Number of GPU")
    parser.add_argument("-u", "--unet_type", dest="unet_type", metavar="U", type=str, default="v1", help="UNet type is v1/v2/v3")
    parser.add_argument("-e", "--epochs", metavar="E", type=int, default=10000, help="Number of epoches", dest="epochs")
    parser.add_argument("-b", "--batch_size", metavar="B", type=int, nargs="?", default=2, help="batch size", dest="batchsize")
    parser.add_argument("-l", "--learning_rate", metavar="L", type=float, nargs="?", default=0.1, help="learning rate", dest="lr")

    parser.add_argument("-f", "--load", dest="load", type=str, default=False, help="load model from a .pth file")
    parser.add_argument("-s", "--scale", dest="scale", type=float, default=0.5, help="scale factor of the images when preprocessing")
    parser.add_argument("-v", "--validation", dest="val", type=float, default=10.0, help="percentage of validation data (0 - 100)")

    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = get_args()
    gpu_id = args.gpu_id
    unet_type = args.unet_type

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device {}".format(device))

    # change here to adapt to your data
    #   n_channels = 3 for RGB images
    #   n_classes is the number of probabilities you want to get per pixel
    #       for 1 class and background, use 1
    #       for 2 classes, use 1
    #       for N > 2 classes, use N
    if unet_type == "v2":
        pass
    if unet_type == "v3":
        net = UNetV3(n_channels=3, n_classes=1)
    else:
        net = UNet(n_channels=3, n_classes=1)

    logging.info("Network:\n\t{} input channels\n\t{} output channels\n".format(net.n_channels, net.n_classes))

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info("Model loaded from {}".format(args.load))

    net.to(device=device)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    try:
        train_net(unet_type=unet_type, net=net, epochs=args.epochs, batch_size=args.batchsize, lr=args.lr,
                  device=device, img_scale=args.scale, val_percent=args.val / 100)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), "INTERRUPTED.pth")
        logging.info("Saved interrupt")
        try:
            sys.exit(0)
        except SystemExit:
            os.exit(0)
