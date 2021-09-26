import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor


class DQN(nn.Module):
    def __init__(self, num_actions: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.fc1 = nn.Linear(1152, 256)  # As we have 64 by 64 images
        self.fc2 = nn.Linear(256, num_actions)
        self.num_actions = num_actions

    def forward(self, experience: Tensor) -> Tensor:
        batch_size = experience.shape[0]
        x = F.relu(self.conv1(experience))
        x = F.relu(self.conv2(x))
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
