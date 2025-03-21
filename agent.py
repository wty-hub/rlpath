import torch
from torch import nn
from torchvision import transforms as T
from PIL import Image
import numpy as np
from pathlib import Path
from collections import deque
import random, datetime, os

from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage

from env import *


class Net(nn.Module):
    def __init__(self, state_shape, action_space_dim):
        super().__init__()
        x = torch.rand(1, *state_shape)
        self.conv = nn.Sequential(
            nn.Conv2d(state_shape[0], 16, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3),
            nn.ReLU(),
            nn.Flatten(),
        )
        x = self.conv(x)
        self.linear = nn.Sequential(
            nn.Linear(x.shape[1], 64),
            nn.ReLU(),
            nn.Linear(64, action_space_dim),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.linear(x)
        return x


class DQNAgent:
    def __init__(
        self, state_shape, n_actions, replay_buffer_size, batch_size, gamma, save_dir
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.state_space = state_shape
        self.action_dim = n_actions
        self.replay_buffer_size = replay_buffer_size
        self.memory = TensorDictReplayBuffer(
            storage=LazyTensorStorage(self.replay_buffer_size, device="cpu")
        )
        self.gamma = gamma

        self.policy_net = Net(state_shape, n_actions).to(self.device)
        self.target_net = Net(state_shape, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)
        self.loss_fn = torch.nn.SmoothL1Loss()
        self.batch_size = batch_size
        self.sync_every = 1000
        self.save_dir = save_dir
        self.epsilon = 1.0
        self.epsilon_decay = 0.99975
        self.epsilon_miminal = 0.05
        self.burnin = self.batch_size  # min. experiences before training
        self.curr_step = 0

    def cache(self, state, next_state, action, reward, done):
        state = torch.from_numpy(state).float()
        next_state = torch.from_numpy(next_state).float()
        action = torch.tensor([action])
        reward = torch.tensor([reward])
        done = torch.tensor([done])
        self.memory.add(
            TensorDict(
                {
                    "state": state,
                    "next_state": next_state,
                    "action": action,
                    "reward": reward,
                    "done": done,
                },
                batch_size=[],
            )
        )

    def recall(self):
        batch = self.memory.sample(self.batch_size).to(self.device)
        state, next_state, action, reward, done = (
            batch.get(key)
            for key in ("state", "next_state", "action", "reward", "done")
        )
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def td_estimate(self, state, action):
        # current_Q = self.policy_net(state).gather(1, action)
        current_Q = self.policy_net(state)[np.arange(0, self.batch_size), action]
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.policy_net(next_state)
        # double DQN
        best_action = torch.argmax(next_state_Q, dim=1)
        # next_Q = self.target_net(next_state).gather(1, best_action)
        next_Q = self.target_net(next_state)[np.arange(0, self.batch_size), best_action]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

    def update_Q_policy(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self):
        checkpoint = {
            "model_state_dict": self.policy_net.state_dict(),
            "epsilon": self.epsilon,
        }
        path = self.save_dir / f"save_{self.curr_step}.chkpt"
        torch.save(checkpoint, path)
        # print("saved to " + path + " at step ", self.curr_step)

    def load(self, path):
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint["model_state_dict"])
        self.sync_Q_target()
        self.epsilon = checkpoint["epsilon"]

    def learn(self):
        if self.curr_step < self.burnin:
            return None, None
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        state, next_state, action, reward, done = self.recall()
        td_est = self.td_estimate(state, action)
        td_tgt = self.td_target(reward, next_state, done)
        loss = self.update_Q_policy(td_est, td_tgt)
        return (td_est.mean().item(), loss)

    def act(self, state):
        if np.random.rand() < self.epsilon:
            action_idx = np.random.randint(self.action_dim)
        else:
            state = torch.tensor(
                state, device=self.device, dtype=torch.float
            ).unsqueeze(0)
            action_values = self.policy_net(state)
            action_idx = torch.argmax(action_values, 1).item()

        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_miminal, self.epsilon)
        self.curr_step += 1
        return action_idx
