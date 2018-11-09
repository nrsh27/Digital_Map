# -*- coding: utf-8 -*-

##############################################################
# scene_context_rrt.py
# Copyright (C) 2018 Tsubasa Hirakawa. All rights reserved.
##############################################################


import math
import numpy as np
import cv2


class SceneContextRRT:

    def __init__(self,
                 input_coordinates,
                 input_feature_map,
                 input_obs_array,
                 input_image,
                 expand_dist=30.0,
                 gamma=500.0,
                 goal_sampling_rate=5,
                 max_iter=10000,
                 verbose=False,
                 animation=False,
                 input_obs_torf=False):

        # flags ============================================
        self.verbose = verbose
        self.animation = animation
        self.obs_torf = input_obs_torf

        # coordinates ======================================
        # sorting [y, x]
        self.coordinates = np.fliplr(input_coordinates)
        # start and goal coordinates
        self.start = self.coordinates[0, :].copy()
        self.end = self.coordinates[-1, :].copy()
        # node and parent [y, x, parent_node_index]
        self.node = np.array([[self.start[0], self.start[1], -1]], dtype=np.int32)
        # cost
        self.cost = np.array([0.0], dtype=np.float32)

        # obstacle map & image for visualize ===============
        self.obstacle_array = input_obs_array.copy()
        self.image_original = input_image.copy()
        self.image = self.image_original.copy()
        # state space size
        self.x_max = self.obstacle_array.shape[1]
        self.y_max = self.obstacle_array.shape[0]

        # feature map ======================================
        self.feature_map = input_feature_map.copy()
        # feature dims.
        self.n_feature = int(self.feature_map.shape[0])
        # weight vector for feature map
        self.weight = np.ones(self.n_feature, dtype=np.float32)
        # scale parameter for regularizer
        self.theta = 1.0
        # cost map
        self.cost_map = np.zeros([self.y_max, self.x_max], dtype=np.float32)
        self.update_weight_cost()

        # RRT_star parameters ==============================
        self.expand_dist = expand_dist
        self.gamma = gamma
        self.goal_sampling_rate = goal_sampling_rate
        self.max_iteration = max_iter

        # for result =======================================
        self.result_node = np.empty([1, 3], dtype=np.int32)
        self.result_path = np.empty([1, 2], dtype=np.int32)
        self.result_path_cost = 0.0
        self.result_image = self.image_original.copy()

    ########################################################
    # methods for planning
    ########################################################
    def planning(self):

        # initialize
        self.node = np.array([[self.start[0], self.start[1], -1]], dtype=np.int32)
        self.cost = np.array([0.0], dtype=np.float32)

        if self.animation:
            # path map
            self.image = self.image_original.copy()
            self.image = cv2.circle(self.image, (self.start[1], self.start[0]), 5, (0, 0, 255), -1)
            self.image = cv2.circle(self.image, (self.end[1], self.end[0]), 5, (0, 255, 255), -1)
            cv2.imshow("path", self.image)

            # cost map
            _cost_map_tmp = self.cost_map.copy()
            _cost_map_tmp = (_cost_map_tmp - np.min(_cost_map_tmp)) / np.max(_cost_map_tmp - np.min(_cost_map_tmp)) * 255
            cv2.imshow("cost", cv2.applyColorMap(_cost_map_tmp.astype(np.uint8), cv2.COLORMAP_JET))
            cv2.waitKey(1)

        _random_samples = self._generate_random_samples()

        for _x_rand in _random_samples:
            _nearest_index = self._nearest_node(_x_rand)

            # _new_node: [y, x, parent]
            _new_node, _new_node_cost, _is_collision = self._steer(_x_rand, _nearest_index)

            if _new_node is None:
                continue

            # check collision and node expansion
            if not _is_collision:
                _near_indices = self._near_nodes(_new_node)

                _new_node, _new_node_cost = self._choose_parent(_new_node, _new_node_cost, _near_indices)
                self.node = np.vstack((self.node, _new_node))
                self.cost = np.hstack((self.cost, _new_node_cost))

                self._rewire(_new_node, _new_node_cost, _near_indices)

                if self.animation:
                    _p = self.node[_new_node[2], :]
                    self.image = cv2.line(self.image, (_new_node[1], _new_node[0]), (_p[1], _p[0]), (255, 255, 255), 1)
                    cv2.imshow("path", self.image)
                    cv2.waitKey(1)

        if self.animation:
            cv2.destroyAllWindows()

    def _generate_random_samples(self):
    # if not self.obs_torf:
        # 画像内すべての範囲でランダムサンプリング
        _random_x = np.random.randint(0, self.x_max, self.max_iteration).astype(np.int32)
        _random_y = np.random.randint(0, self.y_max, self.max_iteration).astype(np.int32)
        _random_samples = np.c_[_random_y, _random_x]
    # else:
    #     # 障害物以外の範囲でランダムサンプリング--------------
    #     random1 = []
    #     for x in range(self.obstacle_array.shape[0]):
    #         for y in range(self.obstacle_array.shape[0]):
    #             if not self.obstacle_array[x][y]:
    #                 random1.append([y, x])
    #     _random1 = np.array(random1)
    #     np.random.shuffle(_random1)
    #     _random_samples = np.ndarray([self.max_iteration, 2]).astype(np.int32)
    #     for i in range(_random_samples.shape[0]):
    #         _random_samples[i] = _random1[i]
    #     # ---------------------------------------------

        _n_goal_sampling = int(float(self.max_iteration) * (float(self.goal_sampling_rate) / 100.0))
        _goal_sampling_index = np.zeros(self.max_iteration, dtype=np.bool)
        _goal_sampling_index[0:_n_goal_sampling] = True
        np.random.shuffle(_goal_sampling_index)
        _random_samples[_goal_sampling_index, :] = self.end

        return _random_samples

    def _nearest_node(self, rand):
        _current_nodes = self.node[:, 0:2]
        _dist = np.linalg.norm(_current_nodes - rand[0:2], axis=1)
        _min_index = np.argmin(_dist)
        return _min_index

    def _near_nodes(self, node):
        _current_nodes = self.node[:, 0:2]
        _dist = np.linalg.norm(_current_nodes - node[0:2], axis=1)
        _r = self.gamma * math.sqrt((math.log(float(self.node.shape[0])) / float(self.node.shape[0])))
        _near_indices = np.where(_dist < _r)
        return _near_indices[0]

    def _steer(self, rand, nearest_index):
        _nearest_node = self.node[nearest_index, :].copy()
        _theta = math.atan2(rand[0] - _nearest_node[0], rand[1] - _nearest_node[1])

        _new_node = _nearest_node.copy()
        _new_node_cost = self.cost[nearest_index].copy()
        _new_node[0] += int(self.expand_dist * math.sin(_theta))
        _new_node[1] += int(self.expand_dist * math.cos(_theta))

        if (_new_node[0] < 0) or (_new_node[0] >= self.y_max) \
                or (_new_node[1] < 0) or (_new_node[1] >= self.x_max):
            return None, 0.0, True

        _new_node[2] = nearest_index

        _line = self._interpolate_line(_nearest_node, _new_node)
        _new_node_cost += self._compute_line_cost(_nearest_node, _new_node, _line)

        _is_collision = self._collision_check(_line)

        return _new_node, _new_node_cost, _is_collision

    def _rewire(self, node, node_cost, near_indices):
        _n_node = self.node.shape[0]

        for i in near_indices:
            _near_node = self.node[i, :]
            _near_cost = self.cost[i]

            _line = self._interpolate_line(node, _near_node)
            _rewire_cost = node_cost + self._compute_line_cost(node, _near_node, _line)

            if _near_cost > _rewire_cost:
                if not self._collision_check(_line):
                    self.node[i, 2] = _n_node - 1
                    self.cost[i] = _rewire_cost

    def _choose_parent(self, node, node_cost, near_indices):

        if len(near_indices) == 0:
            return node, node_cost

        _dist_list = []
        for i in near_indices:
            _line = self._interpolate_line(self.node[i, :], node)

            # no collision
            if not self._collision_check(_line):
                _dist_list.append(self.cost[i] +
                                  self._compute_line_cost(self.node[i, :], node, _line))
            # collision
            else:
                _dist_list.append(1e100)

        _min_cost = min(_dist_list)
        if _min_cost == 1e100:
            if self.verbose:
                print("minimum cost is inf.")
            return node, node_cost
        else:
            _min_index = np.argmin(_dist_list)
            node_cost = _min_cost
            node[2] = near_indices[_min_index]
            return node, node_cost

    def _collision_check(self, line):
        # no collision
        if np.sum(np.logical_not(self.obstacle_array[line[:, 0], line[:, 1]])) == 0:
            return False
        # collision
        else:
            return True

    ########################################################
    # methods for result
    ########################################################
    def result(self):

        # initialize =======================================
        self.result_node = np.empty([1, 3], dtype=np.int32)
        self.result_path = np.empty([1, 2], dtype=np.int32)
        self.result_path_cost = 0.0
        self.result_image = self.image_original.copy()

        # find shortest path node ==========================
        _cost_list = []
        for i in range(self.node.shape[0]):
            _line = self._interpolate_line(self.node[i, :], self.end)
            # no collision
            if not self._collision_check(_line):
                _cost_list.append(self.cost[i] +
                                  self._compute_line_cost(self.node[i], self.end, _line))
            # collision
            else:
                _cost_list.append(1e100)

        _min_cost = min(_cost_list)
        if _min_cost == 1e100:
            import sys
            print("ERROR: cannot find a node connected to goal state.")
            sys.exit(-1)
        _min_index = np.argmin(_cost_list)
        self.result_path_cost = _min_cost

        # result node sequence =============================
        self.result_node = np.r_[self.end, _min_index]

        _current_node_index = _min_index
        while True:
            if _current_node_index == -1:
                break

            self.result_node = np.vstack((self.result_node, self.node[_current_node_index, :]))
            # self.result_node.append(self.nodes[_current_node_index, :])
            _current_node_index = self.node[_current_node_index, 2]

        # self.result_node.reverse()
        self.result_node = np.flipud(self.result_node)

        # result path (coordinate sequence) ================
        self.result_path = self.result_node[0, 0:2]
        for _n in self.result_node[1:, :]:
            if _n[2] == -1:
                continue

            _line = self._interpolate_line(self.node[_n[2], :], _n)
            self.result_path = np.vstack((self.result_path, _line))

        # plot result ======================================
        self.result_image = self.image_original.copy()
        #
        for _n in self.node:
            if _n[2] == -1:
                continue
            # self.result_image = cv2.line(self.result_image,
            #                              (self.node[_n[2], 1], self.node[_n[2], 0]),
            #                              (_n[1], _n[0]),
            #                              (200, 200, 200), 1)
        # 予測結果
        # 予測した経路のノード(節)(青点)
        for _n in self.result_node:
            self.result_image = cv2.circle(self.result_image, (_n[1], _n[0]), 3, (255, 0, 0), -1)
        # 予測した経路の線(青線)
        for i in range(1, len(self.result_node)):
            self.result_image = cv2.line(self.result_image,
                                     (self.result_node[i - 1][1], self.result_node[i - 1][0]),
                                     (self.result_node[i][1], self.result_node[i][0]),
                                     (255, 0, 0), 1)

        # スタート(黄点)
        self.result_image = cv2.circle(self.result_image,
                                   (self.start[1], self.start[0]),
                                   3, (0, 255, 255), -1)
        # ゴール(緑点)
        self.result_image = cv2.circle(self.result_image,
                                   (self.end[1], self.end[0]),
                                   3, (0, 255, 0), -1)

        return self.result_node, self.result_path, self.result_path_cost, self.result_image

    ########################################################
    # methods for feature statistics
    ########################################################
    def compute_empirical_feature_count(self):
        _f_empirical = np.zeros(self.n_feature, dtype=np.float32)
        for i in range(self.n_feature):
            _f_empirical[i] = np.sum(self.feature_map[i, self.coordinates[:, 0], self.coordinates[:, 1]])
        return _f_empirical

    def compute_expected_feature_count(self, input_path):
        _f_expected = np.zeros(self.n_feature, dtype=np.float32)
        for i in range(self.n_feature):
            _f_expected[i] = np.sum(self.feature_map[i, input_path[:, 0], input_path[:, 1]])
        return _f_expected

    ########################################################
    # methods for cost computation
    ########################################################
    def update_weight_cost(self, input_weight=None, input_theta=None):
        # update weight and scale parameter
        if input_weight is not None:
            self.weight = input_weight.copy()
        if input_theta is not None:
            self.theta = input_theta

        # initialize cost map
        self.cost_map *= 0
        # update cost map
        for i in range(self.n_feature):
            self.cost_map += self.weight[i] * self.feature_map[i, :, :]

    def compute_gt_path_cost(self):
        # cost map
        _result_cost = np.sum(self.cost_map[self.coordinates[:, 0], self.coordinates[:, 1]])
        # euclid distance
        _diff = self.coordinates[0:self.coordinates.shape[0] - 1, :] - self.coordinates[1:self.coordinates.shape[0], :]
        _result_cost += self.theta * np.sum(np.sqrt(np.sum(_diff ** 2, axis=1)))
        return _result_cost

    def _compute_line_cost(self, start_node, end_node, line):
        return np.sum(self.cost_map[line[:, 0], line[:, 1]]) + \
                      self.theta * np.linalg.norm(start_node[0:2] - end_node[0:2])

    ########################################################
    # utilities
    ########################################################
    def _interpolate_line(self, start_node, end_node):
        result_line = []

        dx = start_node[1] - end_node[1]
        dy = start_node[0] - end_node[0]

        # the same point
        if dx == 0 and dy == 0:
            result_line.append([start_node[0], start_node[1]])

        # vertical (y direction)
        elif dx == 0:
            for i in self._bi_range(start_node[0], end_node[0]):
                result_line.append([i, start_node[1]])

        # horizontal (x direction)
        elif dy == 0:
            for i in self._bi_range(start_node[1], end_node[1]):
                result_line.append([start_node[0], i])

        # oblique
        else:
            slope = float(dy) / float(dx)
            intercept = start_node[0] - slope * start_node[1]

            # x direction standard
            if abs(slope) <= 1:
                for i in self._bi_range(start_node[1], end_node[1]):
                    result_line.append([int(slope * i + intercept), i])
            # y direction standard
            else:
                for i in self._bi_range(start_node[0], end_node[0]):
                    result_line.append([i, int((i - intercept) / slope)])

        result_line.append([end_node[0], end_node[1]])
        del result_line[0]

        return np.asarray(result_line, dtype=np.int32)

    @staticmethod
    def _bi_range(start, end, step=1):

        if start < end:
            return range(start, end, abs(step))
        else:
            return range(start, end, -abs(step))

    # コストマップ保存用関数
    def generate_cost_map(self):
        # cost map
        _cost_map_tmp = self.cost_map.copy()
        _cost_map_tmp = (_cost_map_tmp - np.min(_cost_map_tmp)) / np.max(_cost_map_tmp - np.min(_cost_map_tmp)) * 255
        return _cost_map_tmp


# for debug
if __name__ == '__main__':

    import time

    print ('障害物の有無を入力(T = 1 or F = 0)')
    input_obs_TorF = input('>>>  ')
    if input_obs_TorF:
        OBS = "obst0acle"
        input_obs_TorF = True
    else:
        OBS = "no_obs"
        input_obs_TorF = False
    coord = np.loadtxt("./intersection_path_prediction/tracking/b00001_0000.txt", dtype=np.int32)
    f_map = np.load("./intersection_path_prediction/feature_map/b00001_feature_map.npy")
    obs_map = np.load("./intersection_path_prediction/obstacle_map/b00001_" + OBS + ".npy")
    image = cv2.imread("../path_data/intersection_path/scene/b00001.png")

    weight = np.loadtxt("sample_weight.txt", dtype=np.float32)
    theta = 1.0

    planner = SceneContextRRT(coord, f_map, obs_map, image,
                              expand_dist=3.0, gamma=500.0,
                              goal_sampling_rate=5, max_iter=1000,
                              verbose=False, animation=True, input_obs_torf=input_obs_TorF)
    planner.update_weight_cost(weight, theta)

    s_time = time.time()
    planner.planning()
    e_time = time.time()
    print(e_time - s_time)

    n, p, c, img = planner.result()
    np.savetxt("./RESULT_DEBUG/node.txt", n)
    np.savetxt("./RESULT_DEBUG/path.txt", p)
    cv2.imwrite("./RESULT_DEBUG/image.png", img)
