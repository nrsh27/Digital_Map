# -*- coding: utf-8 -*-

from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import os
import glob


def erase_newline(s):
    return s.strip()


def read_text(filename):
    with open(filename) as f:
        lines = list(map(erase_newline, f.readlines()))
    return lines


if __name__ == '__main__':
    TRACK_PATH = "../tracking"
    RESULT_image_path = "./image/*.png"
    basename = read_text("../data/basenames.txt")
    png = glob.glob(RESULT_image_path)
    length = 17  # 基準ベクトルの長さ
    ran = 15  # 直進とみなすθの範囲
    list1 = []
    for i, bn in enumerate(basename):
        # coord = np.loadtxt(os.path.join(TRACK_PATH, bn + ".txt"), dtype=np.int32)
        # s = np.array((coord[0][0], coord[0][1], 1))
        # p = np.array((coord[length][0], coord[length][1], 1))
        # g = np.array((coord[-1][0], coord[-1][1], 1))
        #
        # # ---------画像内のベクトル可視化--------- #
        # # print s, p, g
        # # plt.figure()
        # # # 矢印（ベクトル）の始点
        # # X = s[0]
        # # Y = s[1]
        # # # 矢印（ベクトル）の成分
        # # U = p[0] - s[0], g[0] - s[0]
        # # V = p[1] - s[1], g[1] - s[1]
        # # #
        # # # # 矢印（ベクトル）
        # # plt.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1)
        # # #
        # # # # グラフ表示
        # # plt.xlim([0, 500])
        # # plt.ylim([500, 0])
        # # plt.grid()
        # # plt.draw()
        # # plt.show()
        #
        #
        # deg = 0
        # C = np.cos(deg)
        # S = np.sin(deg)
        # tx = -coord[0][0]
        # ty = -coord[0][1]
        # mat = np.array((
        #     (C, -S, tx),
        #     (S, C, ty),
        #     (0, 0, 1)
        # ))
        # s = np.dot(mat, s)
        # p = np.dot(mat, p)
        # g = np.dot(mat, g)
        #
        # # ---------start地点を原点に平行移動したベクトル可視化--------- #
        # # print s, p, g
        # # plt.figure()
        # # # 矢印（ベクトル）の始点
        # # X = s[0]
        # # Y = s[1]
        # # # 矢印（ベクトル）の成分
        # # U = p[0] - s[0], g[0] - s[0]
        # # V = p[1] - s[1], g[1] - s[1]
        # # #
        # # # # 矢印（ベクトル）
        # # plt.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1)
        # # #
        # # # # グラフ表示
        # # plt.xlim([-10, 10])
        # # plt.ylim([10, -10])
        # # plt.grid()
        # # plt.draw()
        # # plt.show()
        #
        # # ---------画像座標系のy軸を逆にしたベクトル可視化--------- #
        # # plt.figure()
        # # # 矢印（ベクトル）の始点
        # # X = s[0]
        # # Y = s[1]
        # # # 矢印（ベクトル）の成分
        # # U = p[0] - s[0], g[0] - s[0]
        # # V = p[1] - s[1], g[1] - s[1]
        # # #
        # # # # 矢印（ベクトル）
        # # plt.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1)
        # # #
        # # plt.xlim([-10, 10])
        # # plt.ylim([-10, 10])
        # # plt.grid()
        # # plt.draw()
        # # plt.show()
        #
        #
        # # print np.rad2deg(np.arctan2(p[1], p[0]))
        # deg = -np.arctan2(p[1], p[0])
        # # if deg > 0:
        # #     deg *= -1
        # C = np.cos(deg)
        # S = np.sin(deg)
        # # print deg, C, S
        # tx = 0
        # ty = 0
        # mat = np.array((
        #     (C, -S, tx),
        #     (S, C, ty),
        #     (0, 0, 1)
        # ))
        # s = np.dot(mat, s)
        # p = np.dot(mat, p)
        # g = np.dot(mat, g)
        # # print s, p, g
        # # print bn, np.rad2deg(np.arctan2(g[1], g[0]))
        #
        # # ---------基準ベクトルがx軸に揃うようにしたベクトル可視化--------- #
        # # plt.figure()
        # # # 矢印（ベクトル）の始点
        # # X = s[0]
        # # Y = s[1]
        # # # 矢印（ベクトル）の成分
        # # U = p[0]-s[0], g[0]-s[0]
        # # V = p[1]-s[1], g[1]-s[1]
        # # #
        # # # # 矢印（ベクトル）
        # # plt.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1)
        # # #
        # # # # グラフ表示
        # # plt.xlim([-10, 10])
        # # plt.ylim([-10, 10])
        # # plt.grid()
        # # plt.draw()
        # # plt.show()
        #
        # theta = np.rad2deg(np.arctan2(g[1], g[0]))
        #
        # if theta < -ran:
        #     print "{:s}_left : {:f}".format(bn, theta)
        #     value = "left"
        # elif theta > ran:
        #     print "{:s}_right : {:f}".format(bn, theta)
        #     value = "right"
        # else:
        #     print "{:s}_straight : {:f}".format(bn, theta)
        #     value = "straight"
        # os.rename(png[i], "./image/" + bn + ".png")
        # os.rename("./image/" + bn + ".png", "./image/" + bn + "_" + value + ".png")

        scene = bn.split('_')
        pnsp = png[i].split('/')
        ps = pnsp[2].split('.')
        # print scene[0],scene[1], pnsp[2], ps[0]
        # os.rename(os.path.join(TRACK_PATH, scene[0] + "_" + scene[1] + ".txt"), os.path.join(TRACK_PATH, ps[0] + ".txt"))

        list1.append(ps[0])
    list2 = []
    train = read_text("../data/test_basenames.txt")
    for bn in train:
        for bas in basename:
            if bn in bas:
                list2.append(bas)
    f = open('../data/test_basenames.txt', 'w')
    for x in list2:
        f.write(str(x) + "\n")
    f.close()
