import gym
import torch
import minerl
import torch.nn as nn

from typing import List, Any

from minecraft.experience_replay import ExperienceReplay
from minecraft.model.dqn import DQN
from minecraft.utils import select_action, eps_decay
from minecraft.transform import preprocess
from minecraft.metrics import gather_metrics


TARGET_NET = DQN(9)


def optimize_Q(
    trainsample: List[Any], model: Any, done: bool, optimizer: Any, gamma: float = 0.99
) -> Any:
    imgs = []
    actions = []
    rewards = []
    for (new_img, action, rew, _) in trainsample:
        imgs.append(new_img)
        actions.append(action)
        rewards.append(torch.tensor(rew))

    y_hat = model(torch.stack(imgs).unsqueeze(1)).sum(1)
    Q_a = torch.stack(rewards)
    if done:
        TARGET_NET.load_state_dict(model.state_dict())
    else:
        Q_a += TARGET_NET(torch.stack(imgs).unsqueeze(1)).sum(1) * gamma
    criterion = nn.MSELoss()
    loss = criterion(y_hat, Q_a)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return model


def train_fn():
    exp_rep = ExperienceReplay(10000)

    env = gym.make("MineRLTreechop-v0")

    done = False

    batch_size = 32
    model = DQN(9)
    eps_start = 1.0
    eps_end = 0.1
    num_steps = 200
    optimizer = torch.optim.RMSprop(model.parameters())

    for episode in range(10):
        initial_obs = preprocess(env.reset()["pov"])  # INITIALIZING SEQUENCE
        obs = initial_obs

        step_num = 0
        ep_rews = []
        ep_attack = []
        while (not done) and (step_num != 1000):
            eps = eps_decay(step_num, eps_start, eps_end, num_steps)
            prev_obs = obs
            action = select_action(obs, eps, model)
            obs, rew, done, _ = env.step(action)

            exp_rep.push((prev_obs, action, rew, preprocess(obs["pov"])))

            obs = preprocess(obs["pov"])

            if len(exp_rep) > batch_size:
                trainsample = exp_rep.sample(batch_size)
                model = optimize_Q(trainsample, model, done, optimizer)
            step_num += 1
            ep_rews.append(rew)
            ep_attack.append(1 if action["attack"] == 1 else 0)
        gather_metrics(ep_rews, ep_attack, episode)


if __name__ == "__main__":
    train_fn()
