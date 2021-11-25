import torch
import torch.nn as nn

from torch.nn import functional as F


class Critic(nn.Module):
    def __init__(self, n_obs, n_actions_contracted) -> None:
        super().__init__()
        self.fc1 = nn.Linear(n_obs, 256)
        self.fc2 = nn.Linear(256+n_actions_contracted, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, states, actions):
        x = F.relu(self.fc1(states))
        x = torch.cat([x, actions], len(actions.shape)-1)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x