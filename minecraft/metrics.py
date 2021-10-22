import numpy as np
import matplotlib.pyplot as plt

from typing import Dict, List


def gather_metrics(rew: List[int], attack: List[int], episode: int) -> None:
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(rew)), rew)
    ax.set_title(f"Reward on episode {episode}")
    fig.savefig(f"/workspace/viz/rew_{episode}.png")

    fig, ax = plt.subplots()
    ax.plot(np.arange(len(attack)), attack)
    ax.set_title(f"Attack number on episode {episode}")
    fig.savefig(f"/workspace/viz/attack_{episode}.png")

    cum_rew = []
    running_sum = 0.0
    for r in rew:
        cum_rew.append(r + running_sum)
        running_sum += r

    fig, ax = plt.subplots()
    ax.plot(np.arange(len(rew)), cum_rew)
    ax.set_title(f"Cummulative reward on episode {episode}")
    fig.savefig(f"/workspace/viz/crew_{episode}.png")

    cum_atk = []
    running_sum = 0.0
    for a in attack:
        cum_atk.append(a + running_sum)
        running_sum += a

    fig, ax = plt.subplots()
    ax.plot(np.arange(len(attack)), cum_atk)
    ax.set_title(f"Cummulative attack on episode {episode}")
    fig.savefig(f"/workspace/viz/cattack_{episode}.png")

    plt.close('all')
