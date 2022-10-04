import numpy as np
import torch
import torch.utils.data
import gym
import os 

path = os.path.dirname (os.path.realpath (__file__))
PACKAGE_PATH = os.path.abspath(os.path.join(path, os.pardir))

class ig_dataset(torch.utils.data.Dataset):
    def __init__(self, args):
        # self.data_ids = [i for i in range(1, 117)] + [i for i in range(118, 149)] + [i for i in range(150, 156)]

        self.args = args

        self.envs = gym.make(args.env_name)

        self.data_buff_states = {}
        self.data_buff_actions = {}

        self.video_idx_list = []
        self.frame_idx_list = []

        self.action_mean = []
        self.action_max = []
        self.action_min = []
        self.steps_size = []

        self.seq_length = 10
        self.state_size = self.envs.obs_robot_len+self.envs.obs_human_len

        self.dataset_name = self.args.gail_experts_dir

        from pathlib import Path
        self.exp_data = torch.load(str(Path(__file__).parent.parent.parent) + '/' + self.dataset_name + '/data.pt')

        # for item in self.data_ids:
        #     self.data_buff_states[item] = self.exp_data['state'][item]
        #     self.data_buff_actions[item] = self.exp_data['action'][item]

            # self.action_mean.append(np.mean(self.data_buff_actions[item], axis=0))
            # self.action_max.append(np.max(self.data_buff_actions[item], axis=0))
            # self.action_min.append(np.min(self.data_buff_actions[item], axis=0))
            # self.steps_size.append(len(self.data_buff_states[item]))

            # self.video_idx_list += [item for i in range(1, len(self.data_buff_states[item]) - self.seq_length - 1)]
            # self.frame_idx_list += [i for i in range(1, len(self.data_buff_states[item]) - self.seq_length - 1)]

        self.data_buff_states = self.exp_data['state']
        self.data_buff_actions = self.exp_data['action']
        self.length = 1
        self.steps_size.append(len(self.data_buff_states))

    def __getitem__(self, index):

        # video_idx = self.video_idx_list[index]
        # frame_idx = self.frame_idx_list[index]
        # padding = np.array([[0.0 for i in range(self.state_size)] for _ in range(self.seq_length - len(inputs))])
        # inputs = np.concatenate(([padding, inputs]), axis=0)

        action_gt = torch.FloatTensor(np.array(self.data_buff_actions[index].cpu().numpy())).view(-1)
        # action_gt[:3] *= 1000.0
        # action_gt[7:10] *= 1000.0

        inputs = torch.FloatTensor(np.array(self.data_buff_states[index].cpu().numpy())).view(-1)

        # vid_id = torch.FloatTensor(np.array([video_idx]))

        return inputs, action_gt, 1

    def __len__(self):
        return self.length