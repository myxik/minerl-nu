import torch
import random
import numpy as np
import torch.nn as nn

from minecraft.model.actor import Actor
from minecraft.model.critic import Critic
from minecraft.experience_replay import ExperienceReplay


class Wolpertinger:
    def __init__(self, n_obs, n_actions, n_actions_contracted, gamma, device, logger):
        self.actor = Actor(n_obs, n_actions)
        self.critic = Critic(n_obs, n_actions_contracted)

        self.actor_target = Actor(n_obs, n_actions)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target = Critic(n_obs, n_actions_contracted)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor.to(device)
        self.critic.to(device)
        self.actor_target.to(device)
        self.critic_target.to(device)

        self.exp_replay = ExperienceReplay(10000)

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.criterion = nn.MSELoss()

        self.device = device
        self.logger = logger

        self.gamma = gamma
        self.n_actions_contracted = n_actions_contracted
        self.n_actions = n_actions

    def optimize_policy(self, trainsample, done, step_num):
        prev_states = []
        actions = torch.empty(size=(len(trainsample), 1))
        rewards = torch.empty(size=(len(trainsample), 1))
        next_states = []

        for i, (prev_state, action, rew, next_state) in enumerate(trainsample):
            prev_states.append(prev_state)
            actions[i] = action
            rewards[i] = rew
            next_states.append(next_state)

        next_states = torch.stack(next_states).view(len(next_states), -1)
        prev_states = torch.stack(prev_states).view(len(prev_states), -1)

        y_hat = self.critic(prev_states.to(self.device), actions.to(self.device))
        y_i = rewards.to(self.device) + self.gamma * self.critic_target(
            next_states.to(self.device), self.proto2topk(self.actor_target(next_states.to(self.device))).to(self.device))

        self.critic_optimizer.zero_grad()
        
        critic_loss = self.criterion(y_hat, y_i)
        critic_loss.backward()

        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()

        actor_loss = -self.critic(prev_states.to(self.device), actions.view(len(actions), -1).to(self.device))
        actor_loss = actor_loss.mean()
        actor_loss.backward()

        self.actor_optimizer.step()

        self.update_targets()
        self.logger.add_scalar("Loss", critic_loss.item(), step_num)
        self.logger.add_scalar("Q_a", y_i.mean().item(), step_num)

    def update_targets(self):
        self.soft_update(self.actor_target, self.actor, 0.2)
        self.soft_update(self.critic_target, self.critic, 0.2)
       
    def soft_update(self, target, source, tau_update):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau_update) + param.data * tau_update
            )

    def proto2topk(self, proto_action):
        _, idx = torch.topk(proto_action, self.n_actions_contracted)
        return idx

    def select_action(self, obs, eps):
        sample = random.random()
        if sample > eps:
            with torch.no_grad():
                obs = obs.view(-1).unsqueeze(0).to(self.device)
                proto = self.actor(obs)

                _, knned = torch.topk(proto, self.n_actions_contracted)

                Q_s = self.critic(obs, knned)

                action = torch.argmax(Q_s, 1)
        else:
            action = np.random.randint(0, self.n_actions)
        return action