#!python3
# -*- coding:utf-8 -*-
# @author: Yize Wang
# @email: wangyize0125@hotmail.com
# @time: 2022/3/30 15:17

import os
import shutil


def create_folder(folder_name: str, remove: bool = True):
    """
    Create a new folder. If the folder exists, remove it first and then make a new one.

    :param folder_name: new folder name
    :param remove: remove the original folder or not
    :return: None
    """

    # this folder exists and remove it
    if os.path.exists(folder_name) and remove:
        shutil.rmtree(folder_name)
        os.mkdir(folder_name)
    # this folder exists but do not remove it
    elif os.path.exists(folder_name) and not remove:
        pass
    # no folder exists
    else:
        os.makedirs(folder_name)


if __name__ == "__main__":
    create_folder("./result")
