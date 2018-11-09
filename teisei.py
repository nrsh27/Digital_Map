# -*- coding: utf-8 -*-

import os
import numpy as np
import cv2
from scene_context_rrt import SceneContextRRT
import sys


def erase_newline(s):
    return s.strip()


def read_text(filename):
    with open(filename) as f:
        lines = list(map(erase_newline, f.readlines()))
    return lines


LABEL_COL = [[255, 0, 0],
             [255, 255, 0],
             [255, 0, 255],
             [0, 255, 0],
             [0, 255, 255],
             [255, 128, 0],
             [255, 0, 128],
             [128, 255, 0],
             [0, 255, 128],
             [128, 0, 255],
             [0, 128, 255]]


RESULT_PATH = "./RESULT_image"
COORD_PATH = "./intersection_path_prediction/tracking"
test_basename = read_text("./intersection_path_prediction/data/test_basenames.txt")

for bn in test_basename:
    coord = np.loadtxt(os.path.join(COORD_PATH, bn + ".txt"), dtype=np.int32)
    scene = bn.split('_')[0]
    IMAGE_PATH = "./intersection_path_prediction/image/" + scene + ".png"
    image = cv2.imread(IMAGE_PATH)
    FEATURE_PATH = "./intersection_path_prediction/feature_map/" + scene + "_feature_map.npy"
    OBSTACLE_PATH = "./intersection_path_prediction/obstacle_map/" + scene + "_no_obs.npy"
    feature_map = np.load(FEATURE_PATH)
    obs_map = np.load(OBSTACLE_PATH)

    # 真値の経路を描画(赤)
    for i in range(1, len(coord)):
        res_image = cv2.line(image, (coord[i - 1][0], coord[i - 1][1]), (coord[i][0], coord[i][1]), (0, 0, 255), 1)
    cnt = 0
    for theta in range(5, 15, 1):
        Theta = theta*0.1

        weight = np.loadtxt("./RESULT/RESULT_theta=" + str(Theta) + "/weight.txt", dtype=np.float32)

        planner = SceneContextRRT(coord, feature_map, obs_map, res_image,
                                  expand_dist=10.0, gamma=500.0,
                                  goal_sampling_rate=5, max_iter=1000,
                                  verbose=False, animation=False, input_obs_torf=False, color=LABEL_COL[cnt])
        planner.update_weight_cost(weight, Theta)
        planner.planning()
        res_node, res_path, res_cost, res_image = planner.result()
        cnt += 1

        cv2.imwrite(os.path.join(RESULT_PATH, bn + ".png"), res_image)

