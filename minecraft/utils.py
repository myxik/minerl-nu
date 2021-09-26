import torch
import random
import numpy as np

from typing import Dict, Any
from torch import Tensor


def select_action(obs: np.ndarray, eps: float, model: Any) -> Dict:
    sample = random.random()
    if sample > eps:
        with torch.no_grad():
            return make_action(model(obs.unsqueeze(0).unsqueeze(1)))
    else:
        return make_dummy_action(model.num_actions)


def eps_decay(
    steps_done: int, eps_start: float, eps_end: float, step_for_decay: int
) -> float:
    if steps_done > step_for_decay:
        return eps_start
    else:
        eps_decay = (eps_start - eps_end) / step_for_decay
        return eps_start - eps_decay * steps_done


def make_action(out: Tensor) -> Dict:
    camera_out = out[:, 2].item()
    sigmoided_out = torch.sigmoid(out).squeeze(0)
    return dict(
        {
            "attack": float(sigmoided_out[0].item() >= 0.5),
            "back": float(sigmoided_out[1].item() >= 0.5),
            "camera": np.array([0, camera_out]),
            "forward": float(sigmoided_out[3].item() >= 0.5),
            "jump": float(sigmoided_out[4].item() >= 0.5),
            "left": float(sigmoided_out[5].item() >= 0.5),
            "right": float(sigmoided_out[6].item() >= 0.5),
            "sneak": float(sigmoided_out[7].item() >= 0.5),
            "sprint": float(sigmoided_out[8].item() >= 0.5),
        }
    )


def make_dummy_action(num_actions: int) -> Dict:
    return dict(
        {
            "attack": random.randint(0, 1),
            "back": random.randint(0, 1),
            "camera": np.array([random.randint(-180, 180), random.randint(-180, 180)]),
            "forward": random.randint(0, 1),
            "jump": random.randint(0, 1),
            "left": random.randint(0, 1),
            "right": random.randint(0, 1),
            "sneak": random.randint(0, 1),
            "sprint": random.randint(0, 1),
        }
    )
