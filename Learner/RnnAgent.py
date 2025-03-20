import torch as th
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical,Normal

class RnnAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RnnAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        if self.args.use_rnn:
            self.rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
        else:
            self.rnn = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.n_actions)
    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.hidden_dim)
        if self.args.use_rnn:
            h = self.rnn(x, h_in)
        else:
            h = F.relu(self.rnn(x))
        q = self.fc2(h)
        return q, h

#Mix_RnnAgent所需的utils
class ScaleShift(nn.Module):
    def __init__(self, scale, shift):
        super().__init__()
        self.scale = scale
        self.shift = shift

    def forward(self, x):
        return x * self.scale + self.shift

class Mix_RnnAgent(nn.Module):
    def __init__(self, input_shape, args, discrete_action_dim, continuous_action_dim=3):
        super(Mix_RnnAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        if self.args.use_rnn:
            self.rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
        else:
            self.rnn = nn.Linear(args.hidden_dim, args.hidden_dim)

        # 离散动作的网络
        self.discrete_action_layer = nn.Linear(args.hidden_dim, discrete_action_dim)

        # 连续动作的网络
        self.continuous_action_mean = nn.Sequential(nn.Linear(args.hidden_dim, continuous_action_dim),
                                                    nn.Tanh())  # 用于确保输出在-1和1之间
        # self.continuous_action_mean = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Linear(args.hidden_dim, 1),
        #         nn.Tanh(),  # 用于确保输出在-1和1之间
        #         ScaleShift(1.5, 2)  # 将输出缩放到0.5到3.5之间
        #     ),
        #     nn.Sequential(
        #         nn.Linear(args.hidden_dim, 1),
        #         nn.Tanh(),  # 用于确保输出在-1和1之间
        #         ScaleShift(15.5, 16.5)  # 将输出缩放到1到32之间
        #     ),
        #     nn.Sequential(
        #         nn.Linear(args.hidden_dim, 1),
        #         nn.Tanh(),  # 用于确保输出在-1和1之间
        #         ScaleShift(14.5,15.5)  # 将输出缩放到1到30之间
        #     ),
        # ])
        self.continuous_action_std = nn.Sequential(
            nn.Linear(args.hidden_dim, continuous_action_dim),
            nn.Softplus()  # 用于确保标准差是正的
        )

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.hidden_dim)
        if self.args.use_rnn:
            h = self.rnn(x, h_in)
        else:
            h = F.relu(self.rnn(x))

        # 离散动作的概率分布
        discrete_probs = self.discrete_action_layer(h)

        # 连续动作的参数
        # continuous_action_mean = th.cat([model(h) for model in self.continuous_action_mean], dim=-1)
        continuous_action_mean = self.continuous_action_mean(h)
        continuous_action_std = self.continuous_action_std(h)
        continuous_params = th.cat([continuous_action_mean, continuous_action_std], dim=-1)

        return th.cat([discrete_probs, continuous_params], dim=-1), h

class NoSharedRnnAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(NoSharedRnnAgent, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.input_shape = input_shape
        if args.action_is_mix:
            self.agents = th.nn.ModuleList([Mix_RnnAgent(input_shape, args,discrete_action_dim=args.n_clients) for _ in range(self.n_agents)])
        else:
            self.agents = th.nn.ModuleList([RnnAgent(input_shape, args) for _ in range(self.n_agents)])
    def init_hidden(self):
        return th.cat([a.init_hidden() for a in self.agents])
    
    def forward(self, inputs, hidden_state):
        hiddens = []
        qs = []
        if inputs.size(0) == self.n_agents:
            for i in range(self.n_agents):
                q, h = self.agents[i](inputs[i].unsqueeze(0), hidden_state[:, i])
                hiddens.append(h)
                qs.append(q)
            return th.cat(qs), th.cat(hiddens).unsqueeze(0)
        else:
            for i in range(self.n_agents):
                inputs = inputs.view(-1, self.n_agents, self.input_shape)
                q, h = self.agents[i](inputs[:, i], hidden_state[:, i])
                hiddens.append(h.unsqueeze(1))
                qs.append(q.unsqueeze(1))
            return th.cat(qs, dim=-1).view(-1, q.size(-1)), th.cat(hiddens, dim=1)

    def cuda(self, device="cuda:0"):
        for a in self.agents:
            a.cuda(device=device)

class EpsilonGreedyActionSelector():

    def __init__(self, args):
        self.args = args

    def select_action(self, agent_inputs, avail_actions, test_mode=False):

        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        
        if test_mode:
            # Greedy action selection only
            self.epsilon = self.args.evaluation_epsilon
        else:
            self.epsilon = self.args.training_epsilon

        # mask actions that are excluded from selection
        masked_q_values = agent_inputs.clone()
        masked_q_values[avail_actions == 0.0] = -float("inf")  # should never be selected!

        random_numbers = th.rand_like(agent_inputs[:, :, 0])
        pick_random = (random_numbers < self.epsilon).long()
        random_actions = Categorical(avail_actions.float()).sample().long()

        picked_actions = pick_random * random_actions + (1 - pick_random) * masked_q_values.max(dim=2)[1]
        return picked_actions

class SoftPoliciesSelector():

    def __init__(self, args):
        self.args = args

    def select_action(self, agent_inputs, test_mode=False):
        m = Categorical(agent_inputs)
        picked_actions = m.sample().long()
        return picked_actions