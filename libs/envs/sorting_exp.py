import os
import sys
import pygame
import libs.envs.cfg as cfg
from libs.envs.modules import *
from itertools import chain
import time as tt
import copy
import numpy as np
import torch
from time import time
import random
from gym import spaces
import gym

class igEnv():
    def __init__(self, args):
        self.args = args
        self.env = gym.make(args.env_name)
        self.step_count = 0

        self.linspace = np.linspace(-0.8, 0.8, 5)
        self.pivot = torch.FloatTensor(np.array([[i, j] for i in self.linspace for j in self.linspace])).view(-1, 2)
        self.pivot_num = len(self.pivot)
        self.pivot_id = 0

        self.pivot_id = (self.pivot_id + 1) % self.pivot_num
        self.random_variable_noise = torch.FloatTensor(np.array([random.uniform(-0.2, 0.2), random.uniform(-0.2, 0.2)])).view(1, 2)
        self.random_variable = self.pivot[self.pivot_id].view(1, 2) + self.random_variable_noise

        self.feature_size = 15
        self.seq_length = 10

        self.padding = torch.FloatTensor(np.array([[0.0 for i in range(self.feature_size)] for _ in range(self.seq_length)]))

        self.observation_space = self.env.observation_space
        self.random_seed_space = spaces.Box(-1.0, 0.0, (2, ))
        self.action_space = self.env.action_space
        self.dataset = self.args.gail_experts_dir
        self.dataset_id = [i for i in range(1, 117)] + [i for i in range(118, 149)] + [i for i in range(150, 156)]

        self.current_step_size = 10
        self.max_step_size = 300
        self.reset_ratio = 0.01

        self.eval_mode = False

    def reset(self):

        self.step_count = 0

        self.pivot_id = (self.pivot_id + 1) % self.pivot_num
        self.random_variable_noise = torch.FloatTensor(np.array([random.uniform(-0.2, 0.2), random.uniform(-0.2, 0.2)])).view(1, 2)
        self.random_variable = self.pivot[self.pivot_id].view(1, 2) + self.random_variable_noise

        self.padding = torch.FloatTensor(np.array([[0.0 for i in range(self.feature_size)] for _ in range(self.seq_length)]))

        return torch.FloatTensor(self.env.reset()[None,:]), self.random_variable

    def start_eval(self, render = False):
        self.eval_mode = True
        self.current_step_size = self.max_step_size
        if render:
            self.env.render()

    def stop_eval(self):
        self.eval_mode = False

    def step(self, actions):
        if torch.is_tensor(actions):
            if actions.device != 'cpu':
                actions = actions[0].detach().cpu().numpy()
        else: actions = actions[0]
        self.step_count += 1
        obs, reward, done, info = self.env.step(actions)

        if not done:
            return torch.FloatTensor(obs[None,:]), torch.FloatTensor([[float(reward)],]), [done], [{}], self.random_variable
        else:
            return torch.FloatTensor(obs[None,:]), torch.FloatTensor([[float(reward)],]), [done], [{'bad_transition':True}], self.random_variable