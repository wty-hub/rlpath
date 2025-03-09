from env import *
from collections import namedtuple
import random
from torch import nn
import torch.nn.functional as F
import torch

# TARGET_UPDATE = 5
# num_episodes = 200
# hidden = 128
# gamma = 0.99
# replay_buffer_size = 100000
# batch_size = 128
# eps_stop = 0.1
# epsilon = 0.6
# eps = 0.5
# Start_epsilon_decaying = 0
# # End_epsilon_decaying = num_episodes // 1
# End_epsilon_decaying = 200
# epsilon_decaying = epsilon / (End_epsilon_decaying - Start_epsilon_decaying)


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
    def __init__(self, state_space, action_space_dim):
        super().__init__()
        t = torch.zeros((1, *state_space))

        self.conv1 = nn.Conv2d(3, 16, 4, 1)
        t = self.conv1(t)

        self.conv2 = nn.Conv2d(16, 32, 4, 1)
        t = self.conv2(t)

        t = t.view(t.shape[0], -1)

        self.fc1 = nn.Linear(t.shape[1], 64)
        self.fc2 = nn.Linear(64, action_space_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # print("x.shape:" + str(x.shape))
        x = x.flatten()
        # print("x.shape after view:" + str(x.shape))

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DQNAgent:
    def __init__(self, state_space, n_actions, replay_buffer_size, batch_size, gamma):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.state_space = state_space
        self.n_actions = n_actions
        self.replay_buffer_size = replay_buffer_size
        self.batch_size = batch_size
        self.gamma = gamma

        self.policy_net = DQN(state_space, n_actions).to(self.device)
        self.target_net = DQN(state_space, n_actions).to(self.device)
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)
        self.memory = ReplayMemory(replay_buffer_size)

    def update_network(self, updates=1):
        for _ in range(updates):
            self._do_network_update()

    def _do_network_update(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        non_final_mask = 1 - torch.tensor(batch.done, dtype=torch.uint8)
        non_final_next_states = [
            s for non_final, s in zip(non_final_mask, batch.next_state) if non_final > 0
        ]
        non_final_next_states = torch.stack(non_final_next_states).to(self.device)
        state_batch = torch.stack(batch.state).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)

        self.optimizer.zero_grad()
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size).to(self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        
        expect_state_action_values = reward_batch + self.gamma * next_state_values

        loss = F.smooth_l1_loss(state_action_values.squeeze(), expect_state_action_values)
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1e-1, 1e-1)
        self.optimizer.step()

    def get_action(self, state, epsilon):
        sample = random.random()
        if sample > epsilon:
            with torch.no_grad():
                state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                q_values = self.policy_net(state).to(self.device)
                return torch.argmax(q_values).item()
        else:
            return random.randrange(self.n_actions)
        
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def store_transition(self, state, action, next_state, reward, done):
        action = torch.Tensor([[action]]).long()
        reward = torch.tensor([reward], dtype=torch.float32)
        next_state = torch.from_numpy(next_state).float()
        state = torch.from_numpy(state).float()
        self.memory.push(state, action, next_state, reward, done)