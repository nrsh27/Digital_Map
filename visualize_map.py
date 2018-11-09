# -*- coding: utf-8 -*-

##############################################################
# make_label.py
# Copyright (C) 2018 Tsubasa Hirakawa. All rights reserved.
##############################################################


import numpy as np
import matplotlib.pyplot as plt
import os


def vis_feature_map(filename):
    feature_map = np.load(filename)

    for i, fmap in enumerate(feature_map):
        plt.figure()
        plt.imshow(fmap, cmap=plt.get_cmap('jet'))
        plt.clim(0, 1)
        plt.tight_layout()
        plt.savefig(scene + "_feature_map-%02d.png" % i)
        plt.close()


def vis_label(filename):
    label = np.load(filename)

    plt.figure()
    plt.imshow(label, cmap=plt.get_cmap('Set1'))
    plt.tight_layout()
    plt.savefig(scene + "_label.png")
    plt.close()


def vis_obstacle(filename):
    obs_map = np.load(filename)
    print np.unique(obs_map)

    plt.figure()
    plt.imshow(obs_map, cmap=plt.get_cmap('gist_gray'))
    plt.tight_layout()
    plt.savefig(scene + "_obstacle_map.png")
    plt.close()


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
        fmap_name = "../feature_map/" + scene + "_feature_map.npy"
        label_name = "../label/" + scene + "_label.npy"
        obs_name = "../obstacle_map/" + scene + "_no_obs.npy"
        if not os.path.exists(fmap_name):
            break
        vis_feature_map(fmap_name)
        vis_label(label_name)
        vis_obstacle(obs_name)

