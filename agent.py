from env import *
from collections import namedtuple
import random
from torch import nn
import torch.nn.functional as F
import torch

TARGET_UPDATE = 5
num_episodes = 200
hidden = 128
gamma = 0.99
replay_buffer_size = 100000
batch_size = 128
eps_stop = 0.1
epsilon = 0.6
eps = 0.5
Start_epsilon_decaying = 0
# End_epsilon_decaying = num_episodes // 1
End_epsilon_decaying = 200
epsilon_decaying = epsilon / (End_epsilon_decaying - Start_epsilon_decaying)


Transition = namedtuple(
    "Transition", ("state", "action", "next_state", "reward", "done")
)


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, state_space_dim, action_space_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 4, 1)
        self.conv2 = nn.Conv2d(16, 32, 4, 1)
        self.fc1 = nn.Linear(64)
        self.fc2 = nn.Linear(64, action_space_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        print("x.shape:" + str(x.shape))
        x = x.flatten()
        print("x.shape after view:" + str(x.shape))

        x = F.relu(self.fc1(x))
        x = self.fc4(x)
        return x