import numpy as np
import matplotlib.pyplot as plt

from typing import Dict, List


def gather_metrics(rew: List[int], episode: int) -> None:
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(rew)), rew)
    ax.set_title(f"Reward on episode {episode}")
    fig.savefig(f"/workspace/viz/rew_{episode}.png")

    cum_rew = []
    running_sum = 0.0
    for r in rew:
        cum_rew.append(r + running_sum)
        running_sum += r

    fig, ax = plt.subplots()
    ax.plot(np.arange(len(rew)), cum_rew)
    ax.set_title(f"Cummulative reward on episode {episode}")
    fig.savefig(f"/workspace/viz/crew_{episode}.png")

    plt.close('all')
