# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import distance_transform_edt

# ラベル
LABEL_COL = [[0, 0, 0],  # 障害物
             [255, 0, 0],  # 道路
             [255, 255, 255],  # 白線
             [0, 255, 0],  # ゼブラゾーン
             [0, 255, 255],  # 横断歩道
             [255, 0, 255],  # 停止線
             [0, 0, 255]]  # 交差点中心
N_LABEL = 7


def exp_dist(img, sigma=5.0):
    """
    Exponentiated distance
    """
    inv_img = np.zeros(img.shape, dtype='float')
    inv_img[img == 0] = 1.0
    dist = distance_transform_edt(inv_img)
    exp_dist = np.exp(-dist/sigma)
    return exp_dist


def img2label(img):
    label = np.zeros([img.shape[0], img.shape[1]], dtype=np.int32)
    B = img[:, :, 0]
    G = img[:, :, 1]
    R = img[:, :, 2]

    # 0〜3のラベルを付ける
    for i, col in enumerate(LABEL_COL):
        if i == 4 or i == 5:
            i = 1
        label[np.logical_and(np.logical_and(B == col[0], G == col[1]), R == col[2])] = i

    # 障害物マップ（True(1):障害物ではない / False(0):障害物）
    obs_map = np.ones(label.shape, dtype=np.bool)

    return label, obs_map


def label2fmap(label):

    feature_map = np.ones([N_LABEL * 8 + 1, label.shape[0], label.shape[1]], dtype=np.float32)

    for i in range(N_LABEL):

        label_map = np.zeros(label.shape, dtype=np.float32)
        label_map[label == i] = 1.0

        feature_map[8 * i + 0, :, :] = label_map.copy()
        feature_map[8 * i + 1, :, :] = 1.0 - label_map.copy()

        label_map = label_map.astype(np.uint8) * 255

        feature_map[8 * i + 2, :, :] = exp_dist(label_map, sigma=5.0)
        feature_map[8 * i + 3, :, :] = exp_dist(label_map, sigma=10.0)
        feature_map[8 * i + 4, :, :] = exp_dist(label_map, sigma=20.0)
        feature_map[8 * i + 5, :, :] = 1.0 - exp_dist(label_map, sigma=5.0)
        feature_map[8 * i + 6, :, :] = 1.0 - exp_dist(label_map, sigma=10.0)
        feature_map[8 * i + 7, :, :] = 1.0 - exp_dist(label_map, sigma=20.0)

    return feature_map


if __name__ == '__main__':
    file_list = []
    for file in os.listdir('../image'):
        if file == '.DS_Store':
            continue
        file_list.append(os.path.splitext(file)[0])
        file_list.sort()
    # シーン名
    SCENE = file_list

    for scene in SCENE:
        # 画像パス
        SCENE_PATH = '../../../path_data/intersection_path/scene_anno/' + scene + '_anno.png'
        if not os.path.exists(SCENE_PATH):
            break
        # label & obstacle map =========================
        anno_img = cv2.imread(SCENE_PATH)
        label, obs_map = img2label(anno_img)
        np.save("../label/" + scene + "_label.npy", label)
        np.save("../obstacle_map/" + scene + "_no_obs.npy", obs_map)

        # feature map ==================================
        label = np.load("../label/" + scene + "_label.npy")
        fmap = label2fmap(label)
        np.save("../feature_map/" + scene + "_feature_map.npy", fmap)
