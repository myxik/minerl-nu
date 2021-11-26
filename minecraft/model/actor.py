import torch
import torch.nn as nn

from torch.nn import functional as F


class Actor(nn.Module):
    def __init__(self, n_obs, n_actions) -> None:
        super().__init__()
        self.fc1 = nn.Linear(n_obs, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 128)
        self.fc5 = nn.Linear(128, n_actions)
        self.activation = nn.Softsign()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.activation(self.fc5(x))
        return x