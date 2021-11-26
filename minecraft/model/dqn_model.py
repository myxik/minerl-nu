import torch
import random
import numpy as np
import torch.nn as nn

from minecraft.model.dqn import DQN


class DQN_model:
    def __init__(self, num_actions, gamma, device, logger):
        self.model = DQN(num_actions)
        self.target_net = DQN(num_actions)
        self.target_net.load_state_dict(self.model.state_dict())
        
        self.device = device
        self.model.to(self.device)
        self.target_net.to(self.device)

        self.logger = logger
        self.optimizer = torch.optim.RMSprop(self.model.parameters())
        
        self.gamma = gamma

    def optimize_policy(self, trainsample, done, step_num, episode):
        prev_states = []
        actions = []
        rewards = []
        next_states = []
        for (prev_state, action, rew, next_state) in trainsample:
            prev_states.append(prev_state)
            actions.append(action)
            rewards.append(torch.tensor(rew))
            next_states.append(next_state)
        y_hat = self.model(torch.stack(prev_states).unsqueeze(1).to(self.device)).sum(1)
        Q_a = torch.stack(rewards).to(self.device)
        if done:
            self.target_net.load_state_dict(self.model.state_dict())
        else:
            Q_a += self.target_net(torch.stack(next_states).unsqueeze(1).to(self.device)).sum(1) * self.gamma
        criterion = nn.HuberLoss()
        loss = criterion(y_hat, Q_a)
        self.optimizer.zero_grad()
        loss.backward()
        self.logger.add_scalar(f"Loss/{episode}", loss.item(), step_num)
        self.logger.add_scalar(f"Q_a/{episode}", Q_a.mean().item(), step_num)
        self.optimizer.step()

    def select_action(self, obs, eps):
        sample = random.random()
        if sample > eps:
            with torch.no_grad():
                action = self.model(obs.unsqueeze(0).unsqueeze(1).to(self.device))
                action = torch.argmax(action, 1)
        else:
            action = np.random.randint(0, 4)
        return action