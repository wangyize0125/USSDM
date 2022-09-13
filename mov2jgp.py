#!python3
# -*- coding:utf-8 -*-
# @author: Yize Wang
# @email: wangyize0125@hotmail.com
# @time: 2022/3/30 15:17

import os
import cv2 as cv
import numpy as np

from file_utils import create_folder
from output_utils import print_func_start, print_info, print_warning


def mov2jpg(mov_file_path: str, output_folder: str = None, start: float = 0, end: float = -1, num_img: int = 30) -> str:
    """
    Movie file to a series of jpg files. jpg filenames are their time instant.
    :param mov_file_path: movie file path
    :param output_folder: output folder for jpg files
    :param start: start time when to start output
    :param end: end time when to stop output
    :param num_img: number of images to output
    :return: output folder name
    """

    print_func_start("Movie ({}) to jpg files".format(mov_file_path))
    mov = cv.VideoCapture(mov_file_path)

    fps = mov.get(cv.CAP_PROP_FPS)
    assert fps > 0.0, "Video /* {} */ not found!".format(mov_file_path)
    num_frame = int(mov.get(cv.CAP_PROP_FRAME_COUNT))
    output_folder = output_folder if output_folder else os.path.dirname(mov_file_path)

    end = end if end > start else 1 / fps * num_frame       # modify end time

    time_instant = np.array(range(num_frame)) / fps
    start_idx, end_idx = np.argwhere(start <= time_instant)[0, 0], np.argwhere(time_instant <= end)[-1, 0]
    time_instant = time_instant[start_idx: end_idx]
    num_img = num_img if num_img >= 1 else time_instant.size
    time_idx = np.int0(np.linspace(0, time_instant.size - 1, num_img, endpoint=True))
    time_instant = time_instant[time_idx].tolist()

    print_info("Video name: {}".format(mov_file_path))
    print_info("\tFPS: {}".format(fps))
    print_info("\tTotal frames: {}".format(num_frame))
    print_info("\tOutput folder: {}".format(output_folder))

    if mov.isOpened():
        create_folder(output_folder, remove=True)

        for i in range(num_frame):
            ret, frame = mov.read()

            if ret and (i / fps) in time_instant:
                filename = os.path.join(output_folder, "{:.4f}.jpg".format(i / fps))

                print("{}th frame read successfully and will be stored in {}".format(i + 1, filename))
                cv.imwrite(filename, frame)
            elif ret and (i / fps) not in time_instant:
                pass
            else:
                print_warning("{}th frame read failed!".format(i + 1))

            if i / fps > end:
                break

    mov.release()

    return output_folder


if __name__ == "__main__":
    mov2jpg("../../videos/MVI_7188.MOV", start=12, end=44)
