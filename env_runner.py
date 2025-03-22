import torch
import itertools
import random
import numpy as np
from functools import partial
from episode_replaybuffer import EpisodeBatch
from utils import select_gpu
from fed import FLServer
from config import fetch_args
from openpyxl import Workbook
class MultiAgentEnv:
    def __init__(self, args,test=False):
        self.test = test

        self.num_agents = args.n_agents
        self.action_space = args.n_actions
        self.observation_space = args.obs_dim
        self.state_space = args.state_dim
        self.episode_limit = args.episode_limit

        self.args = args
        # self.FL_env = [FLServer(args.n_clients,dataset_name) for dataset_name in args.dataset_names]

        #初始化全局状态和观测
        self.obs = np.random.randn(self.num_agents, self.observation_space)
        self.state = np.random.randn(self.state_space)

        self.combinations = list(itertools.product([-1, 0, 1], repeat=3))
        random.shuffle(self.combinations)
        self.combinations = [(-1, -1, 0), (0, 0, 1), (0, 0, 0), (1, -1, 1), (0, -1, 0), (0, -1, -1), (-1, 0, 1), (1, 0, 1),(-1, -1, -1),
                             (1, 1, 0), (1, 0, -1), (1, -1, 0), (0, 1, 0), (-1, 1, 0), (1, -1, -1), (-1, 0, -1),(-1, -1, 1), (-1, 0, 0),
                             (1, 1, -1), (0, 1, -1), (0, 0, -1), (0, 1, 1), (-1, 1, 1), (1, 0, 0), (-1, 1, -1), (0, -1, 1), (1, 1, 1)]

        
    def reset(self):
        if self.test:
            return np.random.randn(self.num_agents, self.observation_space)

        self.FL_env = [FLServer(self.args.n_clients,dataset_name,self.args.non_iid_level) for dataset_name in self.args.dataset_names]
        obs = np.random.randn(self.num_agents, self.observation_space)
        self.last_acc = [0.0, 0.0, 0.0]
        self.delay_lst = [[],[],[]]
        self.energy_lst = [[],[],[]]
        self.comm_overheads = [0,0,0]

        #记录上一次的动作
        self.last_participating = [7, 7, 7]
        self.last_cpu_freq = [1.5, 1.5, 1.5]
        self.last_quantization_bit = [8, 8, 8]
        return obs
    
    def step(self, actions):
        if self.test:
            self.obs = np.random.randn(self.num_agents, self.observation_space)
            self.state = np.random.randn(self.state_space)
            rewards = np.random.randn(self.num_agents,1)
            done = False
            return rewards, done
        actions = actions.cpu().numpy()
        if self.args.action_is_mix:
            num_participating = [int(actions[id][0])+1 for id in range(self.num_agents)]      

            local_epochs = [3, 3, 3]
            batch_size = [32, 32, 32]

            cpu_freq = []
            for id in range(self.num_agents):
                samples = np.random.normal(loc=actions[id][1], scale=0.1, size=5)
                while np.any(samples < 0):
                    samples[samples < 0] = np.random.normal(loc=actions[id][1], scale=0.3, size=np.sum(samples < 0))
                cpu_freq.append((np.round(samples,3)).tolist())

            quantization_bit = []
            for id in range(self.num_agents):
                samples = np.random.normal(loc=actions[id][2], scale=0.25, size=5)
                samples =np.round(samples)
                while np.any(samples <= 0):
                    samples[samples <= 0] = np.random.normal(loc=actions[id][2], scale=0.25, size=np.sum(samples <= 0))
                    samples =np.round(samples)
                quantization_bit.append(samples.tolist())

            BW = [actions[id][3] for id in range(self.num_agents)]
            total = sum(BW)
            if total > 30:
                BW = [np.round(30* (i / total),2) for i in BW]
        else:           
            local_epochs = [3, 3, 3]
            batch_size = [256, 256, 256]

            change_per_agent = [self.combinations[int(actions[id])]for id in range(self.num_agents)]
            print(f"actions_change:{change_per_agent}")
            num_participating = [self.last_participating[id]+change_per_agent[id][0] for id in range(self.num_agents)]
            #检查num_participating是否在合理范围内（1-5）
            if (self.last_acc[2]) <=0.5:
                num_participating = [min(max(3,num_participating[id]),self.args.n_clients) for id in range(self.num_agents)]
            else:
                num_participating = [min(max(2,num_participating[id]),self.args.n_clients) for id in range(self.num_agents)]
            self.last_participating = num_participating

            cpu_freq = [self.last_cpu_freq[id]+change_per_agent[id][1]*0.5 for id in range(self.num_agents)]
            #检查cpu_freq是否在合理范围内（0.5-3.5）
            cpu_freq = [min(max(0.5,cpu_freq[id]),3.5) for id in range(self.num_agents)]
            self.last_cpu_freq = cpu_freq[:]
            #采样
            for id in range(self.num_agents):
                samples = np.random.normal(loc=cpu_freq[id], scale=0.2, size=self.args.n_clients)
                while np.any(samples < 0):
                    samples[samples < 0] = np.random.normal(loc=cpu_freq[id], scale=0.2, size=np.sum(samples < 0))
                cpu_freq[id] = (np.round(samples,3)).tolist()
            
            quantization_bit = [self.last_quantization_bit[id]+change_per_agent[id][2]*2 for id in range(self.num_agents)]
            #检查quantization_bit是否在合理范围内（3-16）
            if (self.last_acc[2]) <=0.5:
                quantization_bit = [min(max(4,quantization_bit[id]),16) for id in range(self.num_agents)]
            else:
                quantization_bit = [min(max(2,quantization_bit[id]),16) for id in range(self.num_agents)]
            self.last_quantization_bit = quantization_bit[:]
            #采样
            for id in range(self.num_agents):
                samples = np.random.normal(loc=quantization_bit[id], scale=0.15, size=self.args.n_clients)
                samples =np.round(samples)
                while np.any(samples <= 0):
                    samples[samples <= 0] = np.random.normal(loc=quantization_bit[id], scale=0.15, size=np.sum(samples <= 0))
                    samples =np.round(samples)
                quantization_bit[id] = samples.tolist()

            
            BW_origin = [self.last_quantization_bit[id] for id in range(self.num_agents)]
            total = sum(BW_origin)
            # if total > 30:
            BW = [np.round(30* (i / total),2) for i in BW_origin]

        quan_schem=[]
        obs_pre = []
        rwd_pre = []
        state_loss=[]
        state_acc=[]
        com_overheads_total = 0
        for server_id in range(self.num_agents):
            fl_iter,quan_bit_avg,com_overheads,server_outputs2 = self.FL_env[server_id].train(1,num_participating[server_id],
                                                                                              local_epochs[server_id],
                                                                                              batch_size[server_id],
                                                                                              quantization_bit[server_id],
                                                                                              cpu_freq[server_id],
                                                                                              BW[server_id])
            obs_individual = np.concatenate((np.array([fl_iter,com_overheads,BW[server_id]]),server_outputs2.reshape(-1))) 
            obs_pre.append(obs_individual)
            
            self.comm_overheads[server_id] = com_overheads

            self.delay_lst[server_id].append(server_outputs2[-2])
            self.energy_lst[server_id].append(server_outputs2[-1])
            rwd_pre.append(-5*server_outputs2[-2]/max(self.delay_lst[server_id]) 
                           - 5*server_outputs2[-1]/max(self.energy_lst[server_id]))

            state_loss.append(server_outputs2[2])
            state_acc.append(server_outputs2[3])

            quan_schem.append(quan_bit_avg)
            com_overheads_total += com_overheads
        self.last_acc = state_acc
        self.state = np.concatenate((np.array([fl_iter,com_overheads_total]),quan_schem,BW,state_loss,state_acc))  # 2+3+3+3+3=14
        
        self.obs = np.array([np.concatenate((t,quan_schem)) for t in obs_pre])  #shape (agent_num, 12)

        #如果state_acc增长，则赋值5
        delta_acc = np.array([5 if state_acc[id] > self.last_acc[id] else 0 for id in range(self.num_agents)])
        # rewards = np.add([ac*10+d_ac for ac,d_ac in zip(state_acc,delta_acc)],
        #                         [2.5*num_participating[id]-quan_schem[id] for id in range(self.num_agents)])
        # rewards = 2*rwd_pre
        rewards = np.add(np.add([ac*20 for ac,ls in zip(state_acc,state_loss)],
                                [quan_schem[id]-self.comm_overheads[id]*50 if id !=2 else quan_schem[id]-self.comm_overheads[id]/1.5 for id in range(self.num_agents)]),
                                rwd_pre)   # acc_up + (B/B_total * quan_bit_avg*num_clients)/(com_overheads_total) + rwd_pre
        rewards = np.clip(rewards, -75, 75)
        
        done = False
        
        return np.reshape(rewards,(self.num_agents,1)), done, BW_origin
    
    def get_obs(self):
        return self.obs
    
    def get_state(self):
        return self.state
    
#定义EpisodeRunner类
class EpisodeRunner:
    def __init__(self, env):
        self.env = env
        self.episode_limit = env.episode_limit
        self.batch_size = 1
        self.device = select_gpu()
        # self.device = "cpu"

        self.t = 0

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.device)
        self.mac = mac
    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def run(self, test_mode=False,excel_dir=None):
        # wb = Workbook()
        # ws1 = wb.active
        self.reset()
        terminated = False
        episode_return = 0

        self.mac.init_hidden(batch_size=self.batch_size)

        while not terminated:
            state_mean = np.mean(self.env.get_state())
            state_std = np.std(self.env.get_state())
            obs_mean = np.mean(self.env.get_obs())
            obs_std = np.std(self.env.get_obs())
            state_norm = (self.env.get_state() - state_mean) / state_std
            obs_norm = (self.env.get_obs() - obs_mean) / obs_std
            pre_transition_data = {
                "state": [state_norm],
                "obs": [obs_norm],
            }
            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the obs to the agents
            ranges = [(0.5, 3), (1, 32), (1, 30)]
            actions = self.mac.select_actions(self.batch, t_ep=self.t, test_mode=test_mode)
            actions_env = actions.detach().clone()
            if self.mac.args.action_is_mix:
                for i in range(actions.shape[-1]-1):
                    min_value, max_value = ranges[i]
                    actions_env[..., i+1] = actions_env[..., i+1] * (max_value - min_value) / 2 + (max_value + min_value) / 2
                    actions_env[..., i+1] = torch.clip(actions_env[..., i+1], min_value, max_value)

            reward, terminated,origin_band = self.env.step(actions_env[0])
            episode_return += reward

            terminated = True if self.t == self.episode_limit else False
            post_transition_data = {
                "actions": actions,
                "reward": [reward],
                "terminated": [(terminated,)]  # Terminate the episode if the time limit is reached
            }
            if test_mode:
                 print(f"Episode return: {episode_return}")
            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1
        if test_mode:
            for server_id in range(3):
                self.env.FL_env[server_id].save_metrics_to_excel(excel_dir,origin_band)
        return self.batch
    


if __name__ == '__main__':
    args = fetch_args()
    env = MultiAgentEnv(args)
    env.reset()
    env.step([0, 1, 2])