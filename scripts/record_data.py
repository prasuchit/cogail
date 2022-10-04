import os
import numpy as np
import torch
path = os.path.dirname (os.path.realpath (__file__))
PACKAGE_PATH = os.path.abspath(os.path.join(path, os.pardir))

import sys
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))


from configs.exp1_config import get_args as get_args_exp1
from configs.exp2_config import get_args as get_args_exp2
from configs.exp3_config import get_args as get_args_exp3
from configs.assistive_config import get_args as get_args_exp4

from libs.envs.env_exp1 import gameEnv
from libs.datasets.dataset_exp1 import Game_dataset
from libs.envs.env_exp2 import igEnv as igEnv_exp2
from libs.datasets.dataset_exp2 import ig_dataset as ig_dataset_exp2
from libs.envs.env_exp3 import igEnv as igEnv_exp3
from libs.datasets.dataset_exp3 import ig_dataset as ig_dataset_exp3
from libs.envs.assistive_exp import igEnv as igEnv_exp4

from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage

import matplotlib.pyplot as plt
from tqdm import tqdm
import pybullet as p
import torch as th
import warnings
warnings.filterwarnings("ignore")

def obs_as_tensor(obs, device='cpu'):
    obs = th.tensor(obs).float().to(device)
    if len(obs.shape) == 2:
        return obs[None, :]
    elif len(obs.shape) == 3:
        return obs


def main(sysargv):
    if sysargv[2] == 'cogail_exp1_2dfq':
        args = get_args_exp1()
    elif sysargv[2] == 'cogail_exp2_handover':
        args = get_args_exp2()
    elif sysargv[2] == 'assistive_gym:FeedingSawyerHuman-v0':
        args = get_args_exp4()
    else:
        args = get_args_exp3()

    if args.env_name == 'cogail_exp1_2dfq':
        envs = gameEnv(args)
    elif args.env_name == 'cogail_exp2_handover':
        envs = igEnv_exp2(args)
    elif args.env_name == 'assistive_gym:FeedingSawyerHuman-v0':
        envs = igEnv_exp4(args)
    else:
        envs = igEnv_exp3(args)

    device = torch.device("cuda:0" if args.cuda else "cpu")

    folder = PACKAGE_PATH + f'/trained_models/{args.env_name}'

    actor_critic, obs_rms = torch.load(os.path.join("{0}/{1}.pt".format(folder, args.env_name)), map_location=device)
    obs, random_seed = envs.reset()
    obs, random_seed = obs.to(device), random_seed.to(device)

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
    num_steps = 100
    num_processes=1

    recurrent_hidden_states = torch.zeros(
        num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    masks = torch.zeros(num_processes, 1, device=device)

    for step in tqdm(range(num_steps)):
        with torch.no_grad():
            _, action, _, recurrent_hidden_states = actor_critic.act(
                                                        obs,
                                                        random_seed,
                                                        recurrent_hidden_states,
                                                        masks,
                                                        deterministic=True)

        new_obs, rewards, dones, infos, random_seed = envs.step(action)
        new_obs, random_seed = new_obs.to(device), random_seed.to(device)
        # print(rewards, actions, obs)
        rewards = sum(rewards)
        dones = all(dones)

        states_rollout.append(obs)
        next_states_rollout.append(new_obs)
        actions_rollout.append(action)
        rewards_rollout.append(rewards)
        dones_rollout.append([dones])
        infos_rollout.append(infos)

        if dones:
            obs, random_seed = envs.reset()
            obs, random_seed = obs.to(device), random_seed.to(device)

            length_stats.append(length)
            reward_stats.append(reward)

            length = 0
            reward = 0
        else:
            obs = new_obs

            length += 1
            reward += rewards
                
        # states_rollout = torch.tensor(states_rollout).float()
        # next_states_rollout = torch.tensor(next_states_rollout).float()
        # actions_rollout = torch.tensor(actions_rollout).float()
        # rewards_rollout = torch.tensor(rewards_rollout).float()
        # dones_rollout = torch.tensor(dones_rollout).float()

    trajectories = {
    'state': states_rollout,
    'action': actions_rollout,
    'reward': rewards_rollout,
    'done': dones_rollout,
    'next_state': next_states_rollout
    }

    save_path = f'{PACKAGE_PATH}/buffers/{args.env_name}'
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    torch.save(trajectories, f'{save_path}/data.pt')    

    print(f'Collect Episodes: {len(length_stats)} | Avg Length: {round(np.mean(length_stats), 2)} | Avg Reward: {th.round(th.mean(th.stack(reward_stats)))}')  

if __name__ == "__main__":
    main(sys.argv)