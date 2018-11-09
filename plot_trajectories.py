# -*- coding: utf-8 -*-

##############################################################
# make_label.py
# Copyright (C) 2018 Tsubasa Hirakawa. All rights reserved.
##############################################################


import os
import numpy as np
import cv2


def remove_newline(s):
    return s.strip()


def plot_trajectories(img_name, basename_file, trajectory_path):
    img = cv2.imread(img_name, 1)

    with open(basename_file) as f:
        basenames = map(remove_newline, f.readlines())

    for bname in basenames:
        if scene in bname:
            path = np.loadtxt(os.path.join(trajectory_path, bname + ".txt"), dtype=np.int32)

            for xy in path:
                img[xy[1], xy[0], :] = (255, 255, 255)

    return img


if __name__ == '__main__':
    file_list = []
    for file in os.listdir('../../../path_data/intersection_path/scene'):
        if file == '.DS_Store':
            continue
        file_list.append(os.path.splitext(file)[0])
        file_list.sort()
    # シーン名
    SCENE = file_list
    for scene in SCENE:
        img_name = "../image/" + scene + ".png"
        basename_file = "../data/basenames.txt"
        trajectory_path = "../tracking"

        img = plot_trajectories(img_name, basename_file, trajectory_path)
        cv2.imwrite(scene + "_trajectories.png", img)


