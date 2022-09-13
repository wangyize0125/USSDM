#!python3
# -*- coding:utf-8 -*-
# @author: Yize Wang
# @email: wangyize0125@hotmail.com
# @time: 2022/4/7 17:20

import os
import glob

import cv2
import math
import cmath
import torch
import traceback
import time as tt
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit

from mov2jgp import mov2jpg
from file_utils import create_folder
from image_utils import is_binary, crop_image, to_grey, to_binary
from output_utils import print_info, print_error, print_func_start, print_warning
from camera_calibration import un_distort_img, calib_video, load_camera_calibrate
from UNet3Plus.unet import UNetV3
from UNet3Plus.predict import predict_and_save


def find_conner(source_file: str, max_iteration: int):
    """
    find rectangular corner in rect image
    :param source_file: filename
    :param max_iteration: maximum interation when refining lines
    :return: corners, [u, v] == [col, row]
    """

    def cal_k_b(point1: np.ndarray, point2: np.ndarray):
        """
        calculate k and b according to two points
        :param point1:
        :param point2:
        :return: k, b
        """

        k = (point2[1] - point1[1]) / (point2[0] - point1[0])
        b = point2[1] - k * point2[0]

        return k, b

    def straight_line(x: np.ndarray, k: float, b: float):
        return k * x + b

    def k_means(line: np.ndarray, points: np.ndarray, max_iter: int):
        """
        k means algorithm for refining lines
        :param line: pre-defined lines
        :param points: inner rectangle points
        :param max_iter: maximum interation
        :return: line k and b
        """

        new_lines = line * 1.0
        for i in range(max_iter):
            dist1 = np.array([np.abs(new_lines[idx, 0] * points[:, 0] - points[:, 1] + new_lines[idx, 1]) / (new_lines[idx, 0] ** 2 + 1) ** 0.5 for idx in range(2)])
            dist2 = np.array([np.abs(new_lines[idx, 0] * points[:, 1] - points[:, 0] + new_lines[idx, 1]) / (new_lines[idx, 0] ** 2 + 1) ** 0.5 for idx in range(2, 4)])
            dist = np.vstack((dist1, dist2))
            min_dist_idx = np.argmin(dist, axis=0)
            # all points on each line
            point_on_line = [points[np.where(min_dist_idx == idx)] for idx in range(new_lines.shape[0])]
            # new fitted line
            current_lines1 = np.array([curve_fit(straight_line, point_on_line[idx][:, 0], point_on_line[idx][:, 1], p0=new_lines[idx])[0] for idx in range(2)])
            current_lines2 = np.array([curve_fit(straight_line, point_on_line[idx][:, 1], point_on_line[idx][:, 0], p0=new_lines[idx])[0] for idx in range(2, 4)])
            current_lines = np.vstack((current_lines1, current_lines2))

            if np.max(np.abs(new_lines - current_lines)) < 1E-5:
                new_lines = current_lines
                print_info("{} / {} rounds are run to refine straight line parameters.".format(i + 1, max_iter))
                break
            else:
                new_lines = current_lines

        return new_lines

    def cal_cross_corners(lines: np.ndarray):
        """
        calculate intersections of the lines
        :param line:
        :return: intersections
        """

        intersections = np.zeros((4, 2))
        for i in range(2):
            intersections[i * 2: (i + 1) * 2, 1] = lines[i, 0] * lines[2: 4, 1] + lines[i, 1] / (1 - lines[i, 0] * lines[2: 4, 0])
            intersections[i * 2: (i + 1) * 2, 0] = lines[2: 4, 0] * intersections[i * 2: (i + 1) * 2, 1] + lines[2: 4, 1]

        return intersections

    def order_corners(corners: np.ndarray):
        theta = [cal_theta(corners[i]) for i in range(corners.shape[0])]
        theta_idx = np.argsort(theta)
        ordered_corner = np.array([corners[idx] for idx in theta_idx])

        return ordered_corner

    def cal_center(corners: np.ndarray):
        """
        calculate centers using four corners
        :param inters: four corners
        :return: center
        """

        k1, b1 = cal_k_b(corners[0], corners[2])
        k2, b2 = cal_k_b(corners[1], corners[3])
        c_x = (b2 - b1) / (k1 - k2)
        c_y = k1 * c_x + b1

        return np.array([c_x, c_y])

    def cal_theta(end: np.ndarray):
        # first area
        if end[0] > 0 and end[1] >= 0:
            return np.arctan(end[1] / end[0])
        elif end[0] <= 0 <= end[1]:
            return np.arccos(end[0] / np.sum(end ** 2) ** 0.5)
        elif end[0] <= 0 and end[1] < 0:
            return 2 * np.pi - np.arccos(end[0] / np.sum(end ** 2) ** 0.5)
        elif end[0] > 0 and end[1] < 0:
            return 2 * np.pi + np.arctan(end[1] / end[0])
        else:
            raise Exception("Unhandled case when calculating theta")

    def find_corner_and_center(bound_points: np.ndarray, cent: np.ndarray, all_cent: np.ndarray):
        kx = (all_cent[1, 1] - all_cent[0, 1]) / (all_cent[1, 0] - all_cent[0, 0] + 1e-10)
        ky = (all_cent[2, 1] - all_cent[0, 1]) / (all_cent[2, 0] - all_cent[0, 0] + 1e-10)
        std_k = np.std(bound_points[:, 0])
        if abs(kx) > abs(ky):
            kx, ky = ky, kx
        else:
            pass

        init_lines = np.array([[kx, -std_k],
                               [kx, std_k],
                               [1 / ky, -std_k],
                               [1 / ky, std_k]])
        refined_lines = k_means(init_lines, bound_points - cent, 500)

        # calculate corners
        rect_corners = cal_cross_corners(refined_lines)
        rect_corners = order_corners(rect_corners) + cent

        return rect_corners

    # load image and find centers of three rectangles
    img = np.array(Image.open(source_file))
    pixel_list, original_centers, edge_points = [80, 160, 240], [], []
    for pixel_value in pixel_list:
        edge_points.append(np.argwhere(img == pixel_value))
        original_centers.append(np.mean(edge_points[-1], axis=0))
    original_centers = np.array(original_centers)

    # find center of mass of the triangle and order rectangle centers anti-clockwise
    center_of_mass = np.mean(original_centers, axis=0)
    centers_regard_com = original_centers - center_of_mass
    thetas = [cal_theta(centers_regard_com[i]) for i in range(3)]
    thetas_idx = np.argsort(thetas)
    ordered_centers = np.array([original_centers[idx] for idx in thetas_idx])

    # find right angle vertices and roll it to the first one
    dists = []
    for i in range(3):
        dists.append(np.sum((ordered_centers[1] - ordered_centers[0]) ** 2) ** 0.5)
        ordered_centers = np.roll(ordered_centers, shift=1, axis=0)
    center_idx = np.argmax(dists) - 1
    ordered_centers = np.roll(ordered_centers, shift=-center_idx, axis=0)
    ordered_edge_points = [0, 0, 0]
    for cluster in edge_points:
        center = np.mean(cluster, axis=0)
        dists = np.sum((ordered_centers - center) ** 2, axis=1) ** 0.5
        center_idx = np.argmin(dists)
        ordered_edge_points[center_idx] = cluster

    # calculate roll dist of rectangle corners according to k of center0 and center 1
    rot_theta = cal_theta((ordered_centers[1] + ordered_centers[2]) / 2 - ordered_centers[0])
    # find corners and centers one-by-one
    all_points = []
    for idx in range(3):
        points = find_corner_and_center(ordered_edge_points[idx], ordered_centers[idx], ordered_centers)
        points_theta = [abs(abs(cal_theta(points[p_idx] - ordered_centers[idx]) - rot_theta) - np.pi) for p_idx in range(4)]
        roll_idx = np.argmin(points_theta)
        points = np.roll(points, shift=-roll_idx, axis=0)

        rect_corner = cal_center(points)
        points = np.vstack((points, rect_corner))
        all_points.append(points)
    all_points = np.vstack(all_points)

    return all_points


def draw_corners(img_file: str, corners: np.ndarray, output: str):
    """
    draw corners on an image
    :param img_file: image filename
    :param corners: corners
    :param output: output file
    :return: None
    """

    img = Image.open(img_file)

    plt.imshow(img)
    plt.scatter(corners[:, 1], corners[:, 0], s=3, c="blue", marker="o")
    for i in range(corners.shape[0] - 1):
        plt.plot([corners[i, 1], corners[i + 1, 1]], [corners[i, 0], corners[i + 1, 0]], "r-", lw=0.5)
    plt.savefig(output)
    plt.clf()


def cal_tran_disp_video(video_file: str, se_cal: tuple, output: str, npz_file: str = "", require_mov2jpg: bool = False):
    """
    calculate displacement using a video file, the first frame is regarded as reference frame
    :param video_file: video file name
    :param se_cal: start and end time for structural displacement calculation
    :param output: output folder name (due to that many folders should be output, a high-level output folder is needed)
    :param npz_file: npz filename, needed when require_calib is False
    :param require_mov2jpg: require mov2jpg for calculation flag
    :return: None
    """

    def update_record(t, d, center_pos):
        """
        record time, acc, vel, and disp in to arrays
        :param t:
        :param d:
        :param center_pos: center of the target
        :return:
        """

        time.append(t)
        disp.append(d)
        pix_disp.append(center_pos)

        if len(time) == 1:
            acc.append([0, 0, 0, 0, 0, 0])
            vel.append([0, 0, 0, 0, 0, 0])

            pix_vel.append([0, 0])
            pix_acc.append([0, 0])
        elif len(time) == 2:
            acc.append([0, 0, 0, 0, 0, 0])
            vel.append((np.array(disp[-1]) - np.array(disp[-2])) / (time[-1] - time[-2]))

            pix_acc.append([0, 0])
            pix_vel.append((np.array(pix_disp[-1]) - np.array(pix_disp[-2])) / (time[-1] - time[-2]))
        else:
            vel.append((np.array(disp[-1]) - np.array(disp[-2])) / (time[-1] - time[-2]))
            acc.append((np.array(vel[-1]) - np.array(vel[-2])) / (time[-1] - time[-2]))

            pix_vel.append((np.array(pix_disp[-1]) - np.array(pix_disp[-2])) / (time[-1] - time[-2]))
            pix_acc.append((np.array(pix_vel[-1]) - np.array(pix_vel[-2])) / (time[-1] - time[-2]))

    def cal_tran_disp(new_rot, new_tran) -> np.ndarray:
        """
        using new rot and new tran to calculate position in the reference frame
        :param new_rot:
        :param new_tran:
        :return:
        """

        # convert obj into camera coordinate
        camera_origin = np.dot(new_rot, np.array([0, 0, 0]).reshape((3, 1))) + new_tran
        # convert camera coordinate into reference coordinate
        ref_origin = np.dot(ref_rot.T, camera_origin) - np.dot(ref_rot.T, ref_tran)

        return ref_origin

    def predict_position():
        predict_vel = np.array(pix_vel[-1]) + np.array(pix_acc[-1]) * (time[-1] - time[-2])
        predict_disp = predict_vel * (time[-1] - time[-2]) + np.array(pix_disp[-1]) - np.array(pix_disp[0])

        return predict_disp[1], predict_disp[0]

    def rotate_matrix_to_euler(rot_matrix: np.ndarray):
        sy = (rot_matrix[0, 0] ** 2 + rot_matrix[1, 0] ** 2) ** 0.5
        singular = sy < 1e-6

        if not singular:
            x = math.atan2(rot_matrix[2, 1], rot_matrix[2, 2])
            y = math.atan2(-rot_matrix[2, 0], sy)
            z = math.atan2(rot_matrix[1, 0], rot_matrix[0, 0])
        else:
            x = math.atan2(-rot_matrix[1, 2], rot_matrix[1, 1])
            y = math.atan2(-rot_matrix[2, 0], sy)
            z = 0

        return np.array([x, y, z])

    print_func_start("Calculate translational displacement using a video file")

    # convert video into images
    if require_mov2jpg:
        mov2jpg(video_file, os.path.join(output, "source_img"), se_cal[0], se_cal[1], num_img=-1)
    else:
        print_info("MOV2JPG flag is false, please ensure {} is not empty".format(os.path.join(output, "source_img")))

    # how many files in source_img file
    img_files = glob.glob(os.path.join(output, "source_img", "*.jpg"))
    time_instant = sorted([float(os.path.splitext(os.path.basename(item))[0]) for item in img_files])
    img_files = ["{:.4f}.jpg".format(item) for item in time_instant]
    print_info("{} images detected from {} sec to {} sec".format(len(img_files), time_instant[0], time_instant[-1]))

    # load unet model
    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
    device = torch.device("cuda")
    unet3plus = UNetV3(n_channels=3, n_classes=1)
    unet3plus.to(device=device)
    unet3plus.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), "UNet3Plus/unet3model.pth"), map_location=device))
    print_info("Load model from {}".format(os.path.join(os.path.dirname(__file__), "UNet3Plus/unet3model.pth")))

    # load camera parameters
    mtx, dist = load_camera_calibrate(npz_file)

    fig = plt.figure(figsize=(8, 8))

    # time, acc, vel, and disp
    time, acc, vel, disp = [], [], [], []
    pix_acc, pix_vel, pix_disp = [], [], []

    create_folder(os.path.join(output, "crop_image"))
    create_folder(os.path.join(output, "unet_predict"))
    create_folder(os.path.join(output, "corners"))
    create_folder(os.path.join(output, "points"))

    file = open(os.path.join(output, "points", "time_series.csv"), "w")
    file.write("t, dx, dy, dz, d_rot, dy_rot, dz_rot, pix_du, pix_dv\n")

    # start calculate structural displacement
    ref_rot, ref_tran, corner, size = None, None, None, None
    ref_corner, ref_size = None, None
    last_rot, last_tran = None, None
    for idx, f_name in enumerate(img_files):
        try:
            f_name_png = f_name[0:-4] + ".png"
            start_time = tt.time()

            out_file = open(os.path.join(output, "points", f_name[0:-4] + ".txt"), "w")

            # first image
            if idx == 0:
                # crop image
                while True:
                    corner_and_size = input("Input corner and size (u: int, v: int, u_size: int, v_size: int): ")
                    data = [int(float(item)) for item in corner_and_size.split(",")]
                    ref_corner = (data[0], data[1])
                    ref_size = (data[2], data[3])
                    corner = [ref_corner[0], ref_corner[1]]
                    size = ref_size
                    corner = list(crop_image(os.path.join(output, "source_img", f_name), corner, size, os.path.join(output, "crop_image", f_name)))

                    is_ok = input("OK? (y/n): ")
                    if is_ok == "y":
                        break
                    else:
                        pass
                print_info("anchor, size: {}, {}".format(corner, size))
            # other image
            else:
                corner = list(crop_image(os.path.join(output, "source_img", f_name), corner, size, os.path.join(output, "crop_image", f_name)))
            # output corner and size
            out_file.write("anchor: {}, {}\n".format(corner[0], corner[1]))
            out_file.write("size: {}, {}\n".format(size[0], size[1]))

            # unet predict
            predict_and_save("v3", unet3plus, Image.open(os.path.join(output, "crop_image", f_name)).resize((512, 512)), device, 1, 0.5,
                             os.path.join(output, "unet_predict", f_name), resize_size=size)

            # find corners
            img_corners = find_conner(os.path.join(output, "unet_predict", f_name_png), 100)
            # move to real position
            img_corners[:, 0] += corner[1]
            img_corners[:, 1] += corner[0]
            # draw corners
            draw_corners(os.path.join(output, "source_img", f_name), img_corners, os.path.join(output, "corners", f_name_png))
            # output corners
            out_file.write("corners:\n")
            for corner_idx in range(img_corners.shape[0]):
                out_file.write("{}, {}\n".format(img_corners[corner_idx, 0], img_corners[corner_idx, 1]))

            # fixed global world coordinate
            obj_corners = np.array([[0, 0, 0], [8, 0, 0], [8, 8, 0], [0, 8, 0], [4, 4, 0],
                                    [10, 0, 0], [18, 0, 0], [18, 8, 0], [10, 8, 0], [14, 4, 0],
                                    [0, 10, 0], [8, 10, 0], [8, 18, 0], [0, 18, 0], [4, 14, 0]],
                                   dtype=np.double) / 100

            # calculate extrinsic parameters
            _, rot, tran, _ = cv2.solvePnPRansac(obj_corners, img_corners, mtx, dist, last_rot, last_tran)
            last_rot, last_tran = rot * 1.0, tran * 1.0
            rot, _ = cv2.Rodrigues(rot)
            # output rot and tran matrix
            out_file.write("rot: {}\n".format(", ".join([str(item) for item in rot.flatten().tolist()])))
            out_file.write("tran: {}\n".format(", ".join([str(item) for item in tran.flatten().tolist()])))

            # for the reference frame, only record ref_rot and ref_tran
            if idx == 0:
                ref_rot, ref_tran = rot, tran

                update_record(float(f_name[0: -4]), [0, 0, 0, 0, 0, 0], img_corners[0].tolist())
            # for the other frames, calculate velocity and acc
            else:
                current_tran = cal_tran_disp(rot, tran)
                current_rot = rotate_matrix_to_euler(np.dot(rot.T, ref_rot))
                # calculate velocity
                update_record(float(f_name[0: -4]), current_tran.flatten().tolist() + current_rot.flatten().tolist(), img_corners[0].tolist())

            # for the other frame, update crop position
            if idx > 0:
                crop_pixel_disp = np.int0(predict_position())
                # output pixel displacement predicted
                out_file.write("pixel displacement predicted: {}, {}\n".format(crop_pixel_disp[0], crop_pixel_disp[1]))

                corner[0] = ref_corner[0] + crop_pixel_disp[0]
                corner[1] = ref_corner[1] + crop_pixel_disp[1]

            end_time = tt.time()
            print_info("handle {}: {} / {}, {:.2f} sec ...".format(f_name[0:-4], idx + 1, len(img_files), end_time - start_time))

            out_file.close()

            # output to file
            file = open(os.path.join(output, "points", "time_series.csv"), "a")
            file.write("{}, {}, {}, {}, {}, {}, {}, {}, {}\n".format(time[-1],
                                                                     disp[-1][0], disp[-1][1], disp[-1][2],
                                                                     disp[-1][3], disp[-1][4], disp[-1][5],
                                                                     pix_disp[-1][1], pix_disp[-1][0]))
            file.close()
        except:
            traceback.print_exc()


if __name__ == "__main__":
    file_name = "MVI_7276"
    cal_tran_disp_video(video_file="C:/wangyize/videos_six/{}.MOV".format(file_name),
                        se_cal=(13.64, 63.64),
                        output="C:/wangyize/calculation_six/{}".format(file_name),
                        npz_file="C:/wangyize/calculation_six/calib/camera_calibration_result.npz",
                        require_mov2jpg=True)
