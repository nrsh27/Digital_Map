# -*- coding: utf-8 -*-

##############################################################
# test.py
# Copyright (C) 2018 Tsubasa Hirakawa. All rights reserved.
##############################################################


if __name__ == '__main__':

    import os
    import numpy as np
    import cv2
    import time
    from scene_context_rrt import SceneContextRRT
    import sys

    argv = sys.argv

    input_theta = float(argv[1])

    def erase_newline(s):
        return s.strip()

    def read_text(filename):
        with open(filename) as f:
            lines = list(map(erase_newline, f.readlines()))
        return lines

    COORD_PATH = "./intersection_path_prediction/tracking"
    RESULT_PATH = "./RESULT/RESULT_theta=" + str(input_theta)

    test_basename = read_text("./intersection_path_prediction/data/test_basenames.txt")

    weight = np.loadtxt(RESULT_PATH + "/weight.txt", dtype=np.float32)
    theta = input_theta

    # print ('障害物の有無を入力(T = 1 or F = 0)')
    # input_obs_TorF = input('>>>  ')
    input_obs_TorF = False
    if input_obs_TorF:
        OBS = "obstacle"
        input_obs_TorF = True
    else:
        OBS = "no_obs"
        input_obs_TorF = False

    # print ('expand_distanceを入力')
    # input_expand_dist = input('>>>  ')
    # print ('max_iterationを入力')
    # input_max_iter = input('>>>  ')

    for bn in test_basename:
        coord = np.loadtxt(os.path.join(COORD_PATH, bn + ".txt"), dtype=np.int32)
        scene = bn.split('_')[0]
        IMAGE_PATH = "./intersection_path_prediction/image/" + scene + ".png"
        FEATURE_PATH = "./intersection_path_prediction/feature_map/" + scene + "_feature_map.npy"
        OBSTACLE_PATH = "./intersection_path_prediction/obstacle_map/" + scene + "_" + OBS + ".npy"
        image = cv2.imread(IMAGE_PATH)
        feature_map = np.load(FEATURE_PATH)
        obs_map = np.load(OBSTACLE_PATH)

        planner = SceneContextRRT(coord, feature_map, obs_map, image,
                                  expand_dist=10.0, gamma=500.0,
                                  goal_sampling_rate=5, max_iter=1000,
                                  verbose=False, animation=False, input_obs_torf=input_obs_TorF)
        planner.update_weight_cost(weight, theta)
        start_time = time.time()
        planner.planning()
        end_time = time.time()
        res_node, res_path, res_cost, res_image = planner.result()

        print(bn, "planning time:", end_time - start_time, "[s]")

        # 真値の経路を描画(赤)
        for i in range(1, len(coord)):
            res_image = cv2.line(res_image, (coord[i-1][0], coord[i-1][1]), (coord[i][0], coord[i][1]), (0, 0, 255), 1)
        # コストマップを保存
        cost_map = planner.generate_cost_map()
        cv2.imwrite("./RESULT/RESULT_theta=" + str(input_theta) + "/" + scene + "_cost_map_theta=" + str(input_theta) + ".png", cv2.applyColorMap(cost_map.astype(np.uint8), cv2.COLORMAP_JET))
        cv2.imwrite(os.path.join(RESULT_PATH, bn + ".png"), res_image)
        np.savetxt(os.path.join(RESULT_PATH, bn + "-node.txt"), res_node, fmt='%d')
        np.savetxt(os.path.join(RESULT_PATH, bn + "-path.txt"), res_path, fmt='%d')
