import gym
import torch
import minerl
import torch.nn as nn

from typing import List, Any
from torch.utils.tensorboard import SummaryWriter
from gym.wrappers import Monitor

from minecraft.experience_replay import ExperienceReplay
from minecraft.model.dqn import DQN
from minecraft.utils import eps_decay
from minecraft.metrics import gather_metrics
from minecraft.wrappers.wrappers import PovWrapper, SimpleActionWrapper, LargeActionWrapper
from minecraft.model.dqn_model import DQN_model
from minecraft.model.wolpertinger import Wolpertinger


def train_fn():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = SummaryWriter()

    env = Monitor(LargeActionWrapper(PovWrapper(gym.make("MineRLTreechop-v0"))), "./video_rep", force=True)
    done = False

    exp_rep = ExperienceReplay(10000)
    batch_size = 128

    eps_start = 1.
    eps_end = 0.001
    num_steps = 2500

    # model = DQN_model(4, 0.99, device, logger)
    model = Wolpertinger(4096, 9, 1, 0.99, device, logger)

    for episode in range(30):
        obs = env.reset()
        done = False

        step_num = 0
        ep_rews = []
        while not done:
            eps = eps_decay(step_num, eps_start, eps_end, num_steps)
            prev_obs = obs
            action = model.select_action(obs, eps)
            obs, rew, done, _ = env.step(action)

            exp_rep.push((prev_obs, action, rew, obs))

            if len(exp_rep) > batch_size:
                trainsample = exp_rep.sample(batch_size)
                model.optimize_policy(trainsample, done, step_num)
            
            step_num += 1
            ep_rews.append(rew)
            logger.add_scalar("Reward/timestep", rew, step_num)
        logger.add_scalar("Reward/episode", sum(ep_rews), episode)
        print(f"Reward is {sum(ep_rews)}")
        gather_metrics(ep_rews, episode)


if __name__ == "__main__":
    train_fn()
