import torch
import torch.nn as nn

from torch.nn import functional as F


class Actor(nn.Module):
    def __init__(self, n_obs, n_actions) -> None:
        super().__init__()
        self.fc1 = nn.Linear(n_obs, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, n_actions)
        self.activation = nn.Softsign()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.activation(self.fc3(x))
        return x