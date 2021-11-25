import cv2
import gym
import torch


class PovWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, obs):
        img = obs["pov"]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return torch.from_numpy(img.copy()).float()


class SimpleActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.base_action = {
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

    def action(self, act):
        if isinstance(act, int):
            act = act
        else:
            act = act.item()

        action = self.base_action.copy()
        if act==0:
            action["forward"] = 0
        if act==1:
            action["jump"] = 1
        if act==2:
            action["camera"] = [0, -30]
        if act==3:
            action["camera"] = [0, 30]
        return action


class LargeActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.base_action = {
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

    def action(self, act):
        if isinstance(act, int):
            act = act
        else:
            act = act.item()

        action = self.base_action.copy()
        if act==0:
            action["forward"] = 0
        if act==1:
            action["jump"] = 1
        if act==2:
            action["camera"] = [0, 30]
        if act==3:
            action["camera"] = [0, -30]
        if act==4:
            action["camera"] = [30, 0]
        if act==5:
            action["camera"] = [-30, 0]
        if act==6:
            action["sprint"] = 1
        if act==7:
            action["camera"] = [0, 90]
        if act==8:
            action["camera"] = [0, -90]
        return action
