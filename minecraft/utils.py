import torch
import random
import numpy as np

from typing import Dict, Any
from torch import Tensor

from minecraft.const import ACTION_SPACE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def select_action(obs: np.ndarray, eps: float, model: Any) -> Dict:
    sample = random.random()
    if sample > eps:
        with torch.no_grad():
            return make_softmax_action(model(obs.unsqueeze(0).unsqueeze(1).to(DEVICE)))
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


def make_sigmoided_action(out: Tensor) -> Dict:
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
    return action_to(random.randint(0, 3))


def make_softmax_action(out: Tensor) -> Dict:
    softmax_out = torch.softmax(out, 1)
    action = torch.argmax(softmax_out, axis=1).item()
    return action_to(action)


def action_to(num: int) -> Dict:
    act = {
        "forward": 1,
        "back": 0,
        "left": 0,
        "right": 0,
        "jump": 0,
        "sneak": 0,
        "sprint": 0,
        "attack" : 1,
        "camera": [0,0],
    }
    if num == 0:
        act['forward'] = 0
    elif num == 1:
        act['jump'] = 1
    elif num == 2:
        act['camera'] = [0, -30]
    elif num == 3:
        act['camera'] = [0, 30]
    elif num == 4:
        act['camera'] = [22.5, 0]
    elif num == 5:
        act['camera'] = [-22.5, 0]
    return act.copy()
