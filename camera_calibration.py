#!python3
# -*- coding:utf-8 -*-
# @author: Yize Wang
# @email: wangyize0125@hotmail.com
# @time: 2022/3/30 15:17

import os
import glob
import time
import cv2 as cv
import numpy as np

from mov2jgp import mov2jpg
from file_utils import create_folder
from output_utils import print_func_start, print_info, print_warning


def calibrate(input_folder: str, num_corners: list or tuple, spacing: float, check_accuracy: bool = True) -> str:
    """
    camera calibrate function using several images
    :param input_folder: input folder where to load images with .jpg extension
    :param num_corners: number of corners in each row and col
    :param spacing: side edge of squares on the calibration board
    :param check_accuracy: check accuracy using the same input images
    :return: camera calibration result file name, a numpy npz file
    """

    print_func_start("Camera calibration")

    objp = np.zeros((num_corners[0] * num_corners[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:num_corners[0], 0:num_corners[1]].T.reshape((-1, 2)) * spacing
    print_info("real world coordinate:\n{}".format(objp))

    obj_points = []         # real world coordinates
    img_points = []         # image coordinates

    images = glob.glob(os.path.join(input_folder, "*.jpg"))
    print_info("detected image files:\n{}".format(images))

    img_size = None

    for f_name in images:
        img = cv.imread(f_name)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)      # to binary image

        img_size = gray.shape[::-1]

        start = time.time()
        ret, corners = cv.findChessboardCorners(gray, num_corners, None)
        end = time.time()

        if ret:
            print_info("Successfully find chess board in {}, cost time: {} min".format(f_name, (end - start) / 60))

            # find chessboard first
            criteria = (cv.TermCriteria_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

            obj_points.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            img_points.append(corners)
            print_info("Successfully refine chess board corners")

            cv.drawChessboardCorners(img, num_corners, corners, ret)
            cv.imwrite(f_name[:-4] + "_calib.png", img)
            cv.destroyAllWindows()
        else:
            print_warning("Failed to find chess board in {}".format(f_name))

    # calibration
    print_info("Image size: {}".format(img_size))
    ret, mtx, dist, r_vec_s, t_vec_s = cv.calibrateCamera(obj_points, img_points, img_size, None, None)

    if check_accuracy:
        mean_error = 0
        for i in range(len(obj_points)):
            img_points2, _ = cv.projectPoints(obj_points[i], r_vec_s[i], t_vec_s[i], mtx, dist)
            error = cv.norm(img_points[i], img_points2, cv.NORM_L2) / len(img_points2)

            mean_error += error
        print_info("total error: {}".format(mean_error / len(obj_points)))

    if ret:
        print_info("mtx (intrinsic parameters): {}".format(mtx))
        print_info("dist (distortion parameters): {}".format(dist))
        print_info("rotation vector (rotational vectors): {}".format(r_vec_s))
        print_info("translation vector (translational vectors): {}".format(t_vec_s))

        np.savez(os.path.join(input_folder, "camera_calibration_result"), mtx=mtx, dist=dist)
        print_info("camera calibration results are stored in {}\\{}".format(input_folder, "camera_calibration_result"))

        return os.path.join(input_folder, "camera_calibration_result.npz")
    else:
        return ""


def load_camera_calibrate(camera_npz_file: str) -> [np.ndarray, np.ndarray]:
    """
    load camera calibration result from npz file and return them
    :param camera_npz_file: npz filename
    :return: mtx and distortion
    """

    print_info("Load camera intrinsic parameters in {}".format(camera_npz_file))
    calibrates = np.load(camera_npz_file)

    return calibrates["mtx"], calibrates["dist"]


def un_distort_img(input_folder: str, camera_npz_file: str, output: str) -> str:
    """
    un distort images under input folder with .jpg file extension
    :param input_folder:
    :param camera_npz_file: camera calibration result
    :param output: output folder name
    :return: mtx corresponding to the un distorted image in a file
    """

    print_func_start("Un distort images in {}".format(input_folder))

    mtx, dist = load_camera_calibrate(camera_npz_file)

    images = glob.glob(os.path.join(input_folder, "*.jpg"))
    print_info("detected image files:\n{}".format(images))

    create_folder(output)

    new_camera_mtx = None
    for f_name in images:
        print_info("Un distort {}".format(f_name))

        img = cv.imread(f_name)
        h, w = img.shape[:2]
        new_camera_mtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

        dst = cv.undistort(img, mtx, dist, None, new_camera_mtx)

        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        cv.imwrite(os.path.join(output, os.path.basename(f_name)), dst)

    new_npz_file = os.path.join(output, "new_mtx")
    np.savez(new_npz_file, mtx=new_camera_mtx)

    return new_npz_file


def calib_video(video: str, out: str, se: tuple or list, num_img: int, num_cor: list or tuple, spacing: float) -> str:
    """
    calibrate camera using a video file, automatically convert video to images and calibrate it
    :param video: video filename
    :param out: out folder
    :param se: start and end time when output img
    :param num_img: number of images
    :param num_cor: corner dimension
    :param spacing: side length of chessboard
    :return: result file name
    """

    # movie to jpg first
    folder = mov2jpg(video, out, se[0], se[1], num_img)
    print_info("JPG files are stored in {}".format(folder))
    # calibrate camera
    result = calibrate(folder, num_cor, spacing, True)
    print_info("NPZ file is at {}".format(result))

    return result


if __name__ == "__main__":
    # npz_file_name = calibrate("../../camon_calibration/MV7135", (8, 8), 0.015)
    # un_distort_img("../../camon_calibration/MV7135", "../../camon_calibration/MV7135/camera_calibration_result.npz")
    npz_file = calib_video("C:/wangyize/videos_six/MVI_7281.MOV", "C:/wangyize/calculation_six/calib", (25, 130), 100, (8, 6), 0.015)
    print(npz_file)
