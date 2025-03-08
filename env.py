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
OBSTACLE = 1

class Env:
    def __init__(self, map_size: Tuple[int, int], init_position: Tuple[int, int], target_positions: List[Tuple[int, int]],
                 obstacle_positions: List[Tuple[int, int]]):
        self.map_size = map_size
        self.init_position = init_position
        self.target_positions = target_positions.copy()
        self.obstacle_positions = obstacle_positions.copy()
        self.grid = np.zeros(self.map_size)
        for obs in obstacle_positions:
            self.grid[obs[0], obs[1]] = OBSTACLE
        
    
    @classmethod
    def load(path: str):
        json