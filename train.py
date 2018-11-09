# -*- coding: utf-8 -*-

##############################################################
# train.py
# Copyright (C) 2018 Tsubasa Hirakawa. All rights reserved.
##############################################################


import os
import math
import numpy as np
import cv2
import multiprocessing as mp
from scene_context_rrt import SceneContextRRT
import sys

argv = sys.argv

input_theta = float(argv[1])

COORD_PATH = "./intersection_path_prediction/tracking"
RESULT_DIR = "./RESULT/RESULT_theta=" + str(input_theta)
CACHE_DIR = "./cache"


# リストの分割
def chunked(iterable, n):
    return [iterable[x:x + n] for x in range(0, len(iterable), n)]


def planning_one_thread(pid, thread_id, basename_list, weight, theta, rrt_params):

    _cost_list = []
    _f_expected_list = []

    for bn in basename_list:
        _coord = np.loadtxt(os.path.join(COORD_PATH, bn + ".txt"), dtype=np.int32)
        scene = bn.split('_')[0]
        # シーン名
        IMAGE_PATH = "./intersection_path_prediction/image/" + scene + ".png"
        FEATURE_PATH = "./intersection_path_prediction/feature_map/" + scene + "_feature_map.npy"
        OBSTACLE_PATH = "./intersection_path_prediction/obstacle_map/" + scene + "_no_obs.npy"
        if not os.path.exists(IMAGE_PATH):
            break
        _image = cv2.imread(IMAGE_PATH)
        _f_map = np.load(FEATURE_PATH)
        _obs_map = np.load(OBSTACLE_PATH)

        _planner = SceneContextRRT(_coord, _f_map, _obs_map, _image,
                                   expand_dist=rrt_params[0], gamma=rrt_params[1],
                                   goal_sampling_rate=rrt_params[2], max_iter=rrt_params[3],
                                   verbose=False, animation=False, input_obs_torf=input_obs_TorF)
        _planner.update_weight_cost(weight, theta)
        _planner.planning()
        _node, _path, _cost, _img = _planner.result()

        _cost_list.append(_cost)
        _f_expected_list.append(_planner.compute_expected_feature_count(_path))

    # save cache
    np.save(os.path.join(CACHE_DIR, "%d-%d-cost.npy" % (pid, thread_id)),
            np.array(_cost_list, dtype=np.float32))
    np.save(os.path.join(CACHE_DIR, "%d-%d-fcount.npy" % (pid, thread_id)),
            np.array(_f_expected_list, dtype=np.float32))



class Trainer:

    def __init__(self,
                 basename_list,
                 expand_dist=30.0,
                 gamma=500.0,
                 goal_sampling_rate=5,
                 max_iter=10000,
                 n_jobs=1,
                 input_obs_torf=False,
                 theta=1.0):
        # flag
        self.obs_torf = input_obs_torf

        self.basename_list = basename_list
        self.n_data = len(self.basename_list)

        # RRT parameters
        self.expand_dist = float(expand_dist)
        self.gamma = float(gamma)
        self.goal_sampling_rate = int(goal_sampling_rate)
        self.max_iter = int(max_iter)
        self.rrt_parameters = (self.expand_dist, self.gamma, self.goal_sampling_rate, self.max_iter)

        # feature map and cost function
        _feature_map = np.load(FEATURE_PATH)
        self.n_feature = _feature_map.shape[0]
        self.weight = np.ones(self.n_feature, dtype=np.float32)
        self.weight_best = self.weight.copy()
        self.theta = theta

        # compute initial cost for ground truth coordinates
        self.max_cost = 1e20
        self.current_cost = 0.0
        self.current_cost = self.compute_cost()
        self.cost_list = [self.current_cost]

        # training
        self.learning_rate = 0.01
        self.DELTA = 10.0
        self.converged = False
        self.f_empirical = np.zeros(self.n_feature, dtype=np.float32)
        self.f_expected = np.zeros(self.n_feature, dtype=np.float32)
        self.f_gradient = np.zeros(self.n_feature, dtype=np.float32)
        self.compute_mean_empirical_feature_count()

        # for multiprocessing
        self.n_jobs = n_jobs
        # use all threads
        if self.n_jobs <= 0 or self.n_jobs >= mp.cpu_count():
            self.n_cpu = mp.cpu_count()
        else:
            self.n_cpu = self.n_jobs
        self.n_data_for_thread = int(math.ceil(float(self.n_data) / float(self.n_cpu)))  # 1スレッド何データか
        self.pid = os.getpid()  # プロセス番号(id)

    def planning(self):

        thread = []
        for thread_id, bn_list in enumerate(chunked(self.basename_list, self.n_data_for_thread)):
            thread.append(mp.Process(target=planning_one_thread,
                                     args=(self.pid, thread_id, bn_list,
                                           self.weight, self.theta, self.rrt_parameters)))
        for t in thread:
            t.start()
        for t in thread:
            t.join()

        self.current_cost = 0.0
        self.f_expected *= 0.0

        for thread_id in range(len(thread)):
            cost_cache = np.load(os.path.join(CACHE_DIR, "%d-%d-cost.npy" % (self.pid, thread_id)))
            f_expected_cache = np.load(os.path.join(CACHE_DIR, "%d-%d-fcount.npy" % (self.pid, thread_id)))

            self.current_cost += np.sum(cost_cache)
            for fc in f_expected_cache:
                self.f_expected += fc

        self.current_cost /= float(self.n_data)
        self.f_expected /= float(self.n_data)

    def gradient_update(self):

        improvement = self.max_cost - self.current_cost
        self.cost_list.append(self.current_cost)
        print("    previous cost:   ", self.max_cost)
        print("    current cost:    ", self.current_cost)
        print("    cost improvement:", improvement)

        if improvement > self.DELTA:
            self.max_cost = self.current_cost
        elif improvement < -self.DELTA:
            print("    warning: cost increases.")
        else:
            improvement = 0

        self.f_gradient = self.f_empirical - self.f_expected

        # improvement
        if improvement > 0:
            print("    improvement: continue to iterate (lr * 2.0)")
            self.weight_best = self.weight.copy()
            self.learning_rate *= 2.0
            self.weight = self.weight_best * np.exp(-self.learning_rate * self.f_gradient)
        # no improvement
        elif improvement < 0:
            print("    no improvement: redo (lr * 0.5)")
            self.learning_rate *= 0.5
            self.weight = self.weight_best * np.exp(-self.learning_rate * self.f_gradient)
        # converged
        else:
            print("    converged.")
            self.converged = True

        print("    lambda:", self.learning_rate)
        print("    f_empirical:", np.vectorize("%.3f".__mod__)(self.f_empirical))
        print("    f_expected:", np.vectorize("%.3f".__mod__)(self.f_expected))
        print("    weight:", np.vectorize("%.3f".__mod__)(self.weight))

    def compute_cost(self):
        _cost = 0.0

        for bn in self.basename_list:
            coord = np.loadtxt(os.path.join(COORD_PATH, bn + ".txt"), dtype=np.int32)
            scene = bn.split('_')[0]
            # シーン名
            IMAGE_PATH = "./intersection_path_prediction/image/" + scene + ".png"
            FEATURE_PATH = "./intersection_path_prediction/feature_map/" + scene + "_feature_map.npy"
            OBSTACLE_PATH = "./intersection_path_prediction/obstacle_map/" + scene + "_no_obs.npy"
            if not os.path.exists(IMAGE_PATH):
                break
            image = cv2.imread(IMAGE_PATH)
            feature_map = np.load(FEATURE_PATH)
            obs_map = np.load(OBSTACLE_PATH)

            planner = SceneContextRRT(coord, feature_map, obs_map, image,
                                      expand_dist=self.expand_dist,
                                      gamma=self.gamma,
                                      goal_sampling_rate=self.goal_sampling_rate,
                                      max_iter=self.max_iter,
                                      verbose=False, animation=False, input_obs_torf=self.obs_torf)
            planner.update_weight_cost(self.weight, self.theta)
            _cost += planner.compute_gt_path_cost()

        _cost /= float(len(self.basename_list))
        return _cost

    def compute_mean_empirical_feature_count(self):

        self.f_empirical *= 0.

        for bn in self.basename_list:
            coord = np.loadtxt(os.path.join(COORD_PATH, bn + ".txt"), dtype=np.int32)
            scene = bn.split('_')[0]
            # シーン名
            IMAGE_PATH = "./intersection_path_prediction/image/" + scene + ".png"
            FEATURE_PATH = "./intersection_path_prediction/feature_map/" + scene + "_feature_map.npy"
            OBSTACLE_PATH = "./intersection_path_prediction/obstacle_map/" + scene + "_no_obs.npy"
            if not os.path.exists(IMAGE_PATH):
                break
            image = cv2.imread(IMAGE_PATH)
            feature_map = np.load(FEATURE_PATH)
            obs_map = np.load(OBSTACLE_PATH)

            planner = SceneContextRRT(coord, feature_map, obs_map, image,
                                      expand_dist=self.expand_dist,
                                      gamma=self.gamma,
                                      goal_sampling_rate=self.goal_sampling_rate,
                                      max_iter=self.max_iter,
                                      verbose=False, animation=False, input_obs_torf=self.obs_torf)

            self.f_empirical += planner.compute_empirical_feature_count()

        self.f_empirical /= self.n_data
        print("mean empirical feature count:")
        print(np.vectorize("%.3f".__mod__)(self.f_empirical))


if __name__ == '__main__':

    def erase_newline(s):
        return s.strip()

    def read_text(filename):
        with open(filename) as f:
            lines = list(map(erase_newline, f.readlines()))
        return lines


    # キャッシュと結果のディレクトリ作成
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    # 学習データ読み込み
    train_basename = read_text("./intersection_path_prediction/data/train_basenames.txt")

    input_obs_TorF = False

    IMAGE_PATH = "./intersection_path_prediction/image/b00001.png"
    FEATURE_PATH = "./intersection_path_prediction/feature_map/b00001_feature_map.npy"
    OBSTACLE_PATH = "./intersection_path_prediction/obstacle_map/b00001_no_obs.npy"

    # 初期化
    trainer = Trainer(train_basename,
                      expand_dist=10.0,
                      gamma=500.0,
                      goal_sampling_rate=5,
                      max_iter=1000,
                      n_jobs=7,
                      input_obs_torf=input_obs_TorF,
                      theta=input_theta)

    print("\ntraining; start\n")
    n_iter = 0
    while True:
        print("%d iteration" % n_iter, "=" * 40)
        trainer.planning()
        trainer.gradient_update()
        np.savetxt(os.path.join(RESULT_DIR, "weight_%04d.txt" % n_iter), trainer.weight_best)
        if trainer.converged:
            break
        n_iter += 1

    print("\ntraining; done\n")
    np.savetxt(os.path.join(RESULT_DIR, "weight.txt"), trainer.weight_best)
    np.savetxt(os.path.join(RESULT_DIR, "cost.txt"), np.array(trainer.cost_list))
