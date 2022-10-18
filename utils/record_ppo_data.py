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

done = False

rewards = []
ep_len = []

states_rollout = []
next_states_rollout = []
actions_rollout = []
rewards_rollout = []
dones_rollout = []
infos_rollout = []

length_stats = []
reward_stats = []

length = 0
reward = 0
num_eps = 1000
curr_eps = 0
num_processes=1

obs = env.reset()

while curr_eps < num_eps:
    action, _states = model.predict(obs, deterministic=True)
    # one_hot_targets = env.get_one_hot(action, env.nAGlobal)
    new_obs, rewards, dones, infos = env.step(action)

    states_rollout.append(obs)
    next_states_rollout.append(new_obs)
    actions_rollout.append(action)
    rewards_rollout.append(rewards)
    dones_rollout.append([dones])
    infos_rollout.append(infos)

    if dones:
        obs = env.reset()

        length_stats.append(length)
        reward_stats.append(reward)

        curr_eps += 1

        length = 0
        reward = 0
    else:
        obs = new_obs

        length += 1
        reward += rewards
                

trajectories = {
'state': states_rollout,
'action': actions_rollout,
'reward': rewards_rollout,
'done': dones_rollout,
'next_state': next_states_rollout
}

save_path = f'{PACKAGE_PATH}/buffers/{env_name}'
if not os.path.isdir(save_path):
    os.makedirs(save_path)
torch.save(trajectories, f'{save_path}/data.pt')    

print(f'Collect Episodes: {len(length_stats)} | Avg Length: {round(np.mean(length_stats), 2)} | Avg Reward: {np.round(np.mean(reward_stats))}')  

env.close()