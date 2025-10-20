import os
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal  # (import is fine even if unused yet)
import numpy as np


class CriticNetwork(nn.Module):
    def __init__(
        self,
        input_dims,                 # e.g., (state_dim,)
        n_actions,
        fc1_dims=256,
        fc2_dims=128,
        name='critic',
        checkpoint_dir='tmp/robo',
        learning_rate=1e-3
    ):
        super(CriticNetwork, self).__init__()

        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_robo')

        # state-action critic: concat(state, action) → Q(s,a)
        state_dim = self.input_dims[0] if isinstance(self.input_dims, (list, tuple)) else self.input_dims
        self.fc1 = nn.Linear(state_dim + n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q1  = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=0.005)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        print(f"Created Critic on {self.device}")
        self.to(self.device)

    def forward(self, state, action):
        # state: (B, state_dim), action: (B, n_actions)
        x = T.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q1 = self.q1(x)   # (B, 1)
        return q1

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file, map_location=self.device))


class ActorNetwork(nn.Module):
    def __init__(
        self,
        input_dims,                 # e.g., (state_dim,)
        fc1_dims=256,
        fc2_dims=128,
        n_actions=2,
        name='actor',
        checkpoint_dir='tmp/robo',
        learning_rate=1e-3
    ):
        super(ActorNetwork, self).__init__()

        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_robo')

        state_dim = self.input_dims[0] if isinstance(self.input_dims, (list, tuple)) else self.input_dims
        self.fc1 = nn.Linear(state_dim, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.out = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        print(f"Created Actor on {self.device}")
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # tanh squashes to [-1, 1]; adjust later with action scaling if needed
        action = T.tanh(self.out(x))
        return action

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file, map_location=self.device))
