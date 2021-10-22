import minerl  # noqa: register MineRL envs as Gym envs.
import gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

from minecraft.wrappers.baseline_wrapper import ObtainPoVWrapper, SerialDiscreteActionWrapper

save_dir = "/workspace/checkpoints"
log_dir = "/workspace/tensorboard_logs"
n_steps = 10000
n_parallel = 16
env_name = "MineRLTreechop-v0"

env = gym.make(env_name)
env = ObtainPoVWrapper(env)
env = SerialDiscreteActionWrapper(env, always_keys=["attack"], reverse_keys=["forward"])

env = make_vec_env(env_id=env, n_envs=n_parallel, seed=2021)

model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=log_dir).learn(total_timesteps=n_steps)

model.save(save_dir)

eval_env = gym.make(env_name)
eval_env = ObtainPoVWrapper(eval_env)
eval_env = SerialDiscreteActionWrapper(eval_env, always_keys=["attack"], reverse_keys=["forward"])

mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)

obs = env.reset()
total_reward = 0
done = False

while not done:
    t = 0
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    total_reward += rewards
    print(f"Reward obtained")
    eval_env.render()
    t += 1
    if done:
        print("Episode finished after {} timesteps".format(t+1))
        break
print(f"Total reward {total_reward}")

env.close()