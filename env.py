import json
import numpy as np
import math
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import namedtuple

import warnings
from typing import *

FREE = 0
OCCUPIED = 1

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

ROBOT_POSITION_IDX = 0
OBSTACLE_IDX = 1
UNVISITED_TARGETS_IDX = 2


class Env:
    def __init__(
        self,
        map_size: Tuple[int, int],
        init_position: Tuple[int, int],
        target_positions: List[Tuple[int, int]],
        obstacle_positions: List[Tuple[int, int]],
    ):
        self.map_size = map_size
        self.init_position = init_position
        self.target_positions = target_positions.copy()
        self.obstacle_positions = obstacle_positions.copy()
        self.grid = np.zeros(self.map_size)
        self.target_num = len(target_positions)

        for obs in obstacle_positions:
            self.grid[obs[0], obs[1]] = OCCUPIED
        self.robot_position = list(self.init_position)
        self.action_space = {0: (0, 1), 1: (1, 0), 2: (0, -1), 3: (-1, 0)}
        self.history = []
        # 0: robot position, 1: obstacles, 2: unvisited targets
        self.state0 = np.zeros((3, map_size[0], map_size[1]))
        self.state0[0, self.robot_position[0], self.robot_position[1]] = 1
        for obs in self.obstacle_positions:
            self.state0[1, obs[0], obs[1]] = 1
        for target in self.target_positions:
            self.state0[2, target[0], target[1]] = 1
        self.cur_state = np.copy(self.state0)

    def reset(self):
        self.robot_position = list(self.init_position)
        self.history.clear()
        self.cur_state = np.copy(self.state0)

    def env_to_json(self) -> str:
        env_dict = {
            "map_size": self.map_size,
            "init_position": self.init_position,
            "target_positions": self.target_positions,
            "obstacle_positions": self.obstacle_positions,
            "grid": self.grid.tolist(),
        }
        return json.dumps(env_dict)

    @classmethod
    def json_to_env(json_str: str):
        env_dict = json.loads(json_str)
        map_size = tuple(env_dict["map_size"])
        init_position = tuple(env_dict["init_position"])
        target_positions = env_dict["target_positions"]
        obstacle_positions = env_dict["obstacle_positions"]
        env = Env(map_size, init_position, target_positions, obstacle_positions)
        env.grid = np.array(env_dict["grid"])
        return env

    def win(self):
        return np.all(self.cur_state[UNVISITED_TARGETS_IDX] == FREE)

    def legal(self, position):
        if self.cur_state[OBSTACLE_IDX, position[0], position[1]] == OCCUPIED:
            return False
        if (
            position[0] < 0
            or position[0] >= self.map_size[0]
            or position[1] < 0
            or position[1] >= self.map_size[1]
        ):
            return False
        return True

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, None]:
        """返回 奖励，是否结束"""
        reward = -1.0
        action = self.action_space[action]
        self.cur_state[
            ROBOT_POSITION_IDX, self.robot_position[0], self.robot_position[1]
        ] = FREE
        self.history.append(action)
        last_position = self.robot_position.copy()
        next_position = (
            self.robot_position[0] + action[0],
            self.robot_position[1] + action[1],
        )
        if not self.legal(next_position):
            reward -= 0.5
        self.robot_position[0] = np.clip(
            self.robot_position[0] + action[0], 0, self.map_size[0]
        )
        self.robot_position[1] = np.clip(
            self.robot_position[1] + action[1], 0, self.map_size[1]
        )
        self.cur_state[
            ROBOT_POSITION_IDX, self.robot_position[0], self.robot_position[1]
        ] = OCCUPIED
        if self.grid[self.robot_position[0]][self.robot_position[1]] == OCCUPIED:
            self.robot_position = last_position

        if (
            self.cur_state[
                UNVISITED_TARGETS_IDX, self.robot_position[0], self.robot_position[1]
            ]
            == OCCUPIED
        ):
            self.cur_state[
                UNVISITED_TARGETS_IDX, self.robot_position[0], self.robot_position[1]
            ] = FREE
            reward += 100.0
            self.cur_state[UNVISITED_TARGETS_IDX]

        if self.win():
            return self.cur_state, reward, True, None
        else:
            return self.cur_state, reward, False, None


def plot_env(env: Env):
    # plt.grid(True)
    plt.imshow(env.grid.T, cmap="gray_r", origin="lower")
    plt.title("Environment Grid")
    plt.xlabel("X")
    plt.ylabel("Y")

    # 标记目标位置
    for target in env.target_positions:
        plt.plot(target[0], target[1], "go")

    # 标记初始位置
    plt.plot(
        env.robot_position[0], env.robot_position[1], "ro", label="Initial Position"
    )

    # plt.legend()
    plt.show()
