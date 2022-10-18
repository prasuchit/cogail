from genericpath import exists
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import torch
import gym
import os 

path = os.path.dirname (os.path.realpath (__file__))
PACKAGE_PATH = os.path.abspath(os.path.join(path, os.pardir))

# env_name = "assistive_gym:FeedingSawyerHuman-v0"
env_name = "ma_gym:HuRoSorting-v0"

env = gym.make(env_name)
model = PPO.load(PACKAGE_PATH + f"/trained_models/{env_name}/{env_name}_ppo", env=env)

# # Evaluate the agent
# mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# print("Mean reward from model: ", mean_reward)

done = False

# env.render()

rewards = []
ep_len = []
length = 0
obs = env.reset()
while len(ep_len) < 100:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    if reward > 1:
        print()
    length += 1
    rewards.append(reward)
    if done:
      obs = env.reset()
      ep_len.append(length)
      length = 0

print(f"{len(ep_len)} epsiodes - Mean reward: {np.mean(rewards)}, Avg length: {np.mean(ep_len)}")
env.close()