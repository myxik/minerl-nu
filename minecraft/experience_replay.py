import random
import numpy as np

from collections import deque
from typing import NamedTuple, List, Any


# EXPERIENCE_TYPE = NamedTuple[np.ndarray, int, np.ndarray, int]


class ExperienceReplay(object):
    def __init__(self, capacity: int) -> None:
        self.memory = deque([], maxlen=capacity)

    def push(self, experience: List[Any]) -> None:
        self.memory.append(experience)

    def sample(self, batch_size: int) -> List[List[Any]]:
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)
