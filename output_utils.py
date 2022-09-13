#!python3
# -*- coding:utf-8 -*-
# @author: Yize Wang
# @email: wangyize0125@hotmail.com
# @time: 2022/3/30 15:17

class DisplayMethod:
    default = str(0)
    highlight = str(1)
    bold = str(2)
    no_bold = str(22)
    underline = str(4)
    no_underline = str(24)
    twinkle = str(5)
    no_twinkle = str(25)
    anti_display = str(7)
    no_anti_display = str(27)

    def __init__(self):
        pass


class ForeColor:
    black = str(30)
    red = str(31)
    green = str(32)
    yellow = str(33)
    blue = str(34)
    magenta = str(35)
    cyan = str(36)
    white = str(37)

    def __init__(self):
        pass


class BackColor:
    black = str(40)
    red = str(41)
    green = str(42)
    yellow = str(43)
    blue = str(44)
    magenta = str(45)
    cyan = str(46)
    white = str(47)

    def __init__(self):
        pass


def print_format(content: str, display_method: str = DisplayMethod.default, fore: str = ForeColor.white, back: str = BackColor.black):
    print("\033[{};{};{}m{}\033[0m".format(display_method, fore, back, content))


def print_func_start(tips: str):
    print("\n")
    print_format(tips, DisplayMethod.underline, ForeColor.red, BackColor.yellow)


def print_warning(warning: str):
    print_format("Warning: {}".format(warning), DisplayMethod.default, ForeColor.red, BackColor.black)


def print_info(info: str):
    print_format("Info: {}".format(info), DisplayMethod.default, ForeColor.blue, BackColor.black)


def print_error(error: str):
    print_format("Error: {}".format(error), DisplayMethod.bold, ForeColor.red, BackColor.black)
