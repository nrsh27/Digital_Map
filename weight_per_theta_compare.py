# -*- coding: utf-8 -*-

# import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

if __name__ == '__main__':
    weight = []
    for i in range(5, 16):
        i *= 0.1
        RESULT_PATH = "./RESULT/RESULT_theta=" + str(i)

        weight.append(np.loadtxt(RESULT_PATH + "/weight.txt", dtype=np.float32))
    df = pd.DataFrame({
        'theta=0.5': weight[0],
        'theta=0.6': weight[1],
        'theta=0.7': weight[2],
        'theta=0.8': weight[3],
        'theta=0.9': weight[4],
        'theta=1.0': weight[5],
        'theta=1.1': weight[6],
        'theta=1.2': weight[7],
        'theta=1.3': weight[8],
        'theta=1.4': weight[9],
        'theta=1.5': weight[10]
    })
    plt.figure()

    labels = {0: "obstacle",
              8: "road",
              16: "white line"}

    for i in range(0, 58, 8):
        df[i:i+8].plot.bar()

        plt.legend(loc='center',
                   bbox_to_anchor=(1.0, 0.5, 0.5, .100),
                   borderaxespad=0., )

        plt.title("Weight per feature map")
        plt.ylabel("weight")

        plt.yscale('log')

        plt.tight_layout()
        # plt.show()
        plt.savefig("./fig/plt_" + str(i) + ".png")

