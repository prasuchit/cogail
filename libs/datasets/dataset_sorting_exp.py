import numpy as np
import torch
import torch.utils.data
import gym
import os 

path = os.path.dirname (os.path.realpath (__file__))
PACKAGE_PATH = os.path.abspath(os.path.join(path, os.pardir))

class Sorting_dataset(torch.utils.data.Dataset):
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
        self.state_size = self.envs.nSGlobal

        self.dataset_name = self.args.gail_experts_dir + '/' + args.env_name

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
        if torch.is_tensor(self.data_buff_actions[index]):
            self.data_buff_actions[index]= torch.FloatTensor(np.array(self.data_buff_actions[index].cpu().numpy(), dtype=np.float32).squeeze()).view(-1)
        if torch.is_tensor(self.data_buff_states[index]):
            self.data_buff_states[index]= torch.FloatTensor(np.array(self.data_buff_states[index].cpu().numpy(), dtype=np.float32).squeeze()).view(-1)

        action_gt = np.array(self.data_buff_actions[index], dtype=np.float32)
        # action_gt[:3] *= 1000.0
        # action_gt[7:10] *= 1000.0

        inputs = np.array(self.data_buff_states[index], dtype=np.float32)

        # vid_id = torch.FloatTensor(np.array([video_idx]))

        return inputs, action_gt, 1

    def __len__(self):
        return self.length


# import numpy as np
# import torch
# import torch.utils.data


# class Sorting_dataset(torch.utils.data.Dataset):
#     def __init__(self, args):
#         self.args = args
#         self.data_ids = [i for i in range(0, 40)] + [i for i in range(100, 140)] + [i for i in range(150, 190)] # dataset 20-20-40-40

#         self.data_buff = {}
#         self.video_idx_list = []
#         self.frame_idx_list = []

#         self.seq_length = 10

#         self.dataset_name = self.args.gail_experts_dir

#         for item in self.data_ids:
#             f = open('{0}/{1}.txt'.format(self.dataset_name, item), 'r')
#             prior_str = f.readline()[:-1]
#             tmp = prior_str.split(' ')
#             self.data_buff[item] = [[float(tmp[0]), float(tmp[1]), float(tmp[2]), float(tmp[3]), float(tmp[4]), float(tmp[5]), float(tmp[6]), float(tmp[7]), float(tmp[8]), float(tmp[9]), float(tmp[10]), float(tmp[11]), float(tmp[12]), float(tmp[13])], ]
#             while True:
#                 tmp_str = f.readline()[:-1]
#                 if tmp_str == '':
#                     break
#                 elif prior_str == tmp_str:
#                     continue
#                 else:
#                     prior_str = tmp_str
#                     tmp = tmp_str.split(' ')
#                     self.data_buff[item].append([float(tmp[0]), float(tmp[1]), float(tmp[2]), float(tmp[3]), float(tmp[4]), float(tmp[5]), float(tmp[6]), float(tmp[7]), float(tmp[8]), float(tmp[9]), float(tmp[10]), float(tmp[11]), float(tmp[12]), float(tmp[13])])

#             # print(self.data_buff[item])

#             self.video_idx_list += [item for i in range(len(self.data_buff[item]) - 1)]
#             self.frame_idx_list += [i for i in range(len(self.data_buff[item]) - 1)]

#         self.length = len(self.video_idx_list)

#         print(self.length, self.length, self.length, self.length)

#     def __getitem__(self, index):

#         video_idx = self.video_idx_list[index]
#         frame_idx = self.frame_idx_list[index]

#         if frame_idx >= (self.seq_length - 1):
#             inputs = np.array(self.data_buff[video_idx][(frame_idx + 1 - self.seq_length):(frame_idx + 1)])[:, :-4]
#         else:
#             inputs = np.array(self.data_buff[video_idx][:(frame_idx + 1)])[:, :-4]
#             padding = np.array([[0.0 for _i in range(10)] for _ in range(self.seq_length - len(inputs))])
#             inputs = np.concatenate(([padding, inputs]), axis=0)

#         if (frame_idx + 1 + self.seq_length) < len(self.data_buff[video_idx]):
#             action_human_gt = torch.FloatTensor(np.array(self.data_buff[video_idx][frame_idx + 1])[-4:-2]).view(-1)
#             action_robot_gt = torch.FloatTensor(np.array(self.data_buff[video_idx][frame_idx + 1])[-2:]).view(-1)
#             action_gt = torch.cat((action_human_gt, action_robot_gt), dim=0)
#         else:
#             action_human_gt = torch.FloatTensor(np.array(self.data_buff[video_idx][frame_idx + 1])[-4:-2]).view(-1)
#             action_robot_gt = torch.FloatTensor(np.array(self.data_buff[video_idx][frame_idx + 1])[-2:]).view(-1)
#             action_gt = torch.cat((action_human_gt, action_robot_gt), dim=0)

#         inputs = torch.FloatTensor(inputs)
#         inputs[:, :8] /= 10.0
#         inputs = inputs.view(-1)

#         vid_id = torch.FloatTensor(np.array([video_idx]))

#         return inputs, action_gt, vid_id

#     def __len__(self):
#         return self.length