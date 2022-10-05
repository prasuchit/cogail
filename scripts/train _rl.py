from genericpath import exists
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

import gym
import os 

path = os.path.dirname (os.path.realpath (__file__))
PACKAGE_PATH = os.path.abspath(os.path.join(path, os.pardir))

env_name = "assistive_gym:FeedingSawyerHuman-v0"

env = gym.make(env_name)

train_iters = 150_000

if exists(PACKAGE_PATH + f"/trained_models/{env_name}/{env_name}_ppo.zip"):
    model = PPO.load(PACKAGE_PATH + f"/trained_models/{env_name}/{env_name}_ppo", env=env)
    model.set_env(env)
    train_iters = 10_000
    model.learn(train_iters)
else:
    model = PPO("MlpPolicy", env_name).learn(train_iters)
    model.set_env(env)

model.save(PACKAGE_PATH + f"/trained_models/{env_name}/{env_name}_ppo")

del model  # delete trained model to demonstrate loading

model = PPO.load(PACKAGE_PATH + f"/trained_models/{env_name}/{env_name}_ppo", env=env)

# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

print("Mean reward: ", mean_reward)