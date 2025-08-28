import torch
import random
import numpy as np
import torch.nn.functional as F
import os
import torch.nn as nn
import torch.optim as optim
import collections
import numpy as np
import math
import gc
from re import X
from tqdm import tqdm
from types import SimpleNamespace
from torch.distributions import Normal
from torch.distributions import Categorical # <-- 导入Categorical

class PolicyNetDiscrete(nn.Module):
    """原生支持离散动作的Actor网络"""
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNetDiscrete, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim) # 输出层维度为离散动作数量

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # 直接输出每个离散动作的logits
        return self.fc3(x)

class QValueNetDiscrete(nn.Module):
    """原生支持离散动作的Critic网络"""
    def __init__(self, state_dim, hidden_dim, action_dim, action_embedding_dim=16):
        super(QValueNetDiscrete, self).__init__()
        # 使用嵌入层来处理离散动作输入
        self.action_embedding = nn.Embedding(action_dim, action_embedding_dim)
        
        self.fc1 = nn.Linear(state_dim + action_embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        # a 是离散动作的索引 (整数)
        a_embedding = self.action_embedding(a.long().squeeze(-1)) # 获取动作的嵌入向量
        cat = torch.cat([x, a_embedding], dim=1) # 拼接状态和动作嵌入
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)

class SACDiscrete(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, 
                 actor_lr, critic_lr, alpha_lr, target_entropy, tau, gamma, device,
                 segments, replica):
        super(SACDiscrete, self).__init__()
        self.action_dim = action_dim
        self.actor = PolicyNetDiscrete(state_dim, hidden_dim, action_dim).to(device)
        self.critic_1 = QValueNetDiscrete(state_dim, hidden_dim, action_dim).to(device)
        self.critic_2 = QValueNetDiscrete(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic_1 = QValueNetDiscrete(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic_2 = QValueNetDiscrete(state_dim, hidden_dim, action_dim).to(device)

        self.replay_buffer = ReplayBuffer(100000)
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float, requires_grad=True, device=device)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        
        self.target_entropy = target_entropy
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.segments = segments
        self.replica = replica
        
        self.policy_advantage = torch.tensor(0, dtype=torch.float).to(self.device)
        self.beta = torch.tensor(0, dtype=torch.float).to(self.device)
        self.actor_param = []

    def get_param(self):
        """获取当前actor网络的参数列表"""
        self.actor_param = [p.data.clone() for p in self.actor.parameters()]

    def take_action(self, state, deterministic=False):
        # state = torch.tensor([state], dtype=torch.float).to(self.device)
        logits = self.actor(state)
        dist = Categorical(logits=logits)
        if deterministic:
            action = dist.probs.argmax()
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item()

    def get_metric(self, model, states, actions, Coff):
        """
        计算费雪信息矩阵(FIM)并动态求解混合系数beta。
        这是原始RSM-MASAC的完整、未经简化的实现。
        """
        # 1. 准备参数差异向量
        self.get_param()
        model.get_param() # model是cache_agent
        param_div = [p_self - p_model for p_self, p_model in zip(self.actor_param, model.actor_param)]
        param_div_vector = torch.cat([p.view(-1) for p in param_div]).unsqueeze(-1)

        # 2. 计算当前策略的对数概率和概率
        logits_self = self.actor(states)
        dist_self = Categorical(logits=logits_self)
        log_prob_self = dist_self.log_prob(actions.squeeze(-1))
        probs = dist_self.probs.gather(1, actions.long()).squeeze(-1) # 获取每个样本对应动作的概率

        # 3. 逐样本计算梯度外积，以构建FIM
        FIM = None
        batch_size = states.size(0)
        
        # 将参数差异向量和相关张量转换为半精度以节省显存
        param_div_vector = param_div_vector.to(dtype=torch.float16)

        for i in range(batch_size):
            self.actor_optimizer.zero_grad()
            # 对单个样本的对数概率进行反向传播
            log_prob_sample = -log_prob_self[i]
            log_prob_sample.backward(retain_graph=True)

            # 收集单个样本的梯度，并转换为半精度
            sample_grads_fp16 = []
            for param in self.actor.parameters():
                if param.grad is not None:
                    sample_grads_fp16.append(param.grad.view(-1).detach().to(dtype=torch.float16))
                    param.grad = None # 清空梯度，为下一个样本做准备
            
            if not sample_grads_fp16: continue

            sample_grad_vector = torch.cat(sample_grads_fp16).unsqueeze(-1)
            
            # 清理CUDA缓存
            torch.cuda.empty_cache()
            
            # 计算梯度外积，并根据该动作的概率加权
            outer_product = probs[i].to(dtype=torch.float16) * sample_grad_vector @ sample_grad_vector.t()
            
            if FIM is None:
                FIM = outer_product
            else:
                FIM.add_(outer_product)

            # 再次清理，释放中间张量
            del outer_product
            gc.collect()
            torch.cuda.empty_cache()

        if FIM is None:
            return torch.tensor(0.0, dtype=torch.float32).to(self.device)

        FIM /= batch_size # FIM是期望值，所以除以batch_size

        # 4. 计算beta值
        # FIM计算结果可能存在数值不稳定的问题，进行一些保护性操作
        fim_term = param_div_vector.t() @ FIM @ param_div_vector
        if fim_term.item() <= 1e-9: # 避免除以零或极小值
            return torch.tensor(0.0, dtype=torch.float32).to(self.device)

        # 确保所有计算都在相同的数据类型上进行
        policy_adv_fp16 = self.policy_advantage.to(dtype=torch.float16)
        coff_fp16 = Coff.to(dtype=torch.float16)
        
        # beta平方项
        beta_squared = (2 * policy_adv_fp16) / (coff_fp16 * fim_term)
        
        # 如果结果为负（由于数值误差），则beta为0
        if beta_squared.item() < 0:
            beta = torch.tensor(0.0, dtype=torch.float32)
        else:
            beta = torch.sqrt(beta_squared).squeeze()
        
        # 转换回float32并返回
        return beta.clone().detach().to(dtype=torch.float32)

    def get_policy_adv(self, model, batch_size, gamma):
        if self.replay_buffer.size() < batch_size:
            self.policy_advantage = torch.tensor(0.0, dtype=torch.float).to(self.device)
            self.beta = torch.tensor(0.0, dtype=torch.float).to(self.device)
            return

        b_s, b_a, b_r, b_ns, b_d, b_a_log = self.replay_buffer.sample(batch_size)
        
        states = torch.tensor(b_s, dtype=torch.float).to(self.device)
        actions = torch.tensor(b_a, dtype=torch.long).view(-1, 1).to(self.device)
        actions_log = torch.tensor(b_a_log, dtype=torch.float).view(-1, 1).to(self.device)

        with torch.no_grad():
            logits_tilde = model.actor(states)
            dist_tilde = Categorical(logits=logits_tilde)
            log_prob_tilde = dist_tilde.log_prob(actions.squeeze(-1)).unsqueeze(-1)
            pi_tilde_prob = torch.exp(log_prob_tilde)
            tilde_entropy = dist_tilde.entropy().mean()

            logits_self = self.actor(states)
            dist_self = Categorical(logits=logits_self)
            log_prob_self = dist_self.log_prob(actions.squeeze(-1)).unsqueeze(-1)
            pi_self_prob = torch.exp(log_prob_self)
            self_entropy = dist_self.entropy().mean()

            q1_value = self.critic_1(states, actions)
            q2_value = self.critic_2(states, actions)
            Q_value = torch.min(q1_value, q2_value)
            
            off_actions_prob = torch.exp(actions_log)

            entropy_minus = tilde_entropy - self_entropy
            policy_adv = ((pi_tilde_prob - pi_self_prob) / (off_actions_prob + 1e-8)) * Q_value + self.log_alpha.exp() * entropy_minus

        self.policy_advantage = torch.mean(policy_adv)
        
        # 如果策略优势小于等于0，则无需计算beta，直接返回
        if self.policy_advantage.item() <= 0:
            self.beta = torch.tensor(0.0, dtype=torch.float).to(self.device)
            return
            
        # 计算Coff系数
        # 注意: 这里的 Q_value 可能范围很大，需要注意数值稳定性
        max_q_val = torch.max(torch.abs(Q_value))
        Coff = (2 * max_q_val * gamma) / ((1 - gamma) ** 2)

        # 调用完整的get_metric来计算beta
        self.beta = self.get_metric(model, states, actions, Coff)

    # ... calc_target, update, soft_update, mix_policy 等其他方法保持不变 ...
    # ... 请确保将上一回答中修正后的这些方法复制到此处 ...
    def calc_target(self, rewards, next_states, dones):
        with torch.no_grad():
            next_logits = self.actor(next_states)
            next_dist = Categorical(logits=next_logits)
            next_probs = next_dist.probs
            
            # 计算下一个状态的期望Q值
            q1_target_next = torch.zeros(next_states.size(0), 1, device=self.device)
            q2_target_next = torch.zeros(next_states.size(0), 1, device=self.device)
            for i in range(self.action_dim):
                action_i = torch.full((next_states.size(0), 1), i, device=self.device)
                q1_target_next += next_probs[:, i].unsqueeze(1) * self.target_critic_1(next_states, action_i)
                q2_target_next += next_probs[:, i].unsqueeze(1) * self.target_critic_2(next_states, action_i)
            
            min_q_target = torch.min(q1_target_next, q2_target_next)
            entropy_term = next_dist.entropy().unsqueeze(1)
            
            td_target = rewards + self.gamma * (min_q_target + self.log_alpha.exp() * entropy_term) * (1 - dones)
        return td_target

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.int64).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(np.array(transition_dict['dones'], dtype=np.uint8), dtype=torch.float).view(-1, 1).to(self.device)

        td_target = self.calc_target(rewards, next_states, dones)
        critic_1_loss = F.mse_loss(self.critic_1(states, actions), td_target.detach())
        critic_2_loss = F.mse_loss(self.critic_2(states, actions), td_target.detach())
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        logits = self.actor(states)
        dist = Categorical(logits=logits)
        probs = dist.probs
        log_probs = F.log_softmax(logits, dim=1)

        with torch.no_grad():
            q1_all_actions = self.critic_1.fc_out(F.relu(self.critic_1.fc2(F.relu(self.critic_1.fc1(torch.cat([states.unsqueeze(1).repeat(1,self.action_dim,1), self.critic_1.action_embedding(torch.arange(self.action_dim, device=self.device)).unsqueeze(0).repeat(states.size(0),1,1)], dim=-1)))))).squeeze(-1)
            q2_all_actions = self.critic_2.fc_out(F.relu(self.critic_2.fc2(F.relu(self.critic_2.fc1(torch.cat([states.unsqueeze(1).repeat(1,self.action_dim,1), self.critic_2.action_embedding(torch.arange(self.action_dim, device=self.device)).unsqueeze(0).repeat(states.size(0),1,1)], dim=-1)))))).squeeze(-1)
            min_q_values = torch.min(q1_all_actions, q2_all_actions)
        
        actor_loss = torch.mean(torch.sum(probs * (self.log_alpha.exp() * log_probs - min_q_values.detach()), dim=1))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        entropy = dist.entropy().mean()
        alpha_loss = -self.log_alpha * (entropy.detach() - self.target_entropy).detach()
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()
        
        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def get_reshaped_param(self):
        return [p.data.clone() for p in self.actor.parameters()]

    def get_segments(self, target_model_weights, p, segments):
        flat_m = torch.cat([w.flatten() for w in target_model_weights])
        seg_length = len(flat_m) // segments
        start_idx = p * seg_length
        end_idx = (p + 1) * seg_length if p < segments - 1 else len(flat_m)
        shape_list = [w.shape for w in target_model_weights]
        return flat_m[start_idx:end_idx], shape_list

    def reconstruct(self, flat_m_list, shape_list):
        flat_m = torch.cat(flat_m_list)
        result = []
        current_pos = 0
        for shape in shape_list:
            num_elements = torch.prod(torch.tensor(shape)).item()
            param = flat_m[current_pos : current_pos + num_elements].view(shape)
            result.append(param)
            current_pos += num_elements
        return result
        
    def cache_param(self, actor_param):
        for p_cache, p_new in zip(self.actor.parameters(), actor_param):
            p_cache.data.copy_(p_new.data)

    def mix_policy(self, actor_param):
        for p_self, p_new in zip(self.actor.parameters(), actor_param):
            p_self.data.copy_((1 - self.beta) * p_self.data + self.beta * p_new.data)
        for k, x in zip(actor_param, self.actor.parameters()):
            x.data.copy_(k.data)

def clip_grad_norm(parameters, max_norm, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad.data.mul_(clip_coef)
    return total_norm

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity) 

    def add(self, state, action, reward, next_state, done, action_log): 
        self.buffer.append((state, action, reward, next_state, done, action_log)) 

    def sample(self, batch_size): 
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, action_log = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done, action_log

    def size(self): 
        return len(self.buffer)

class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(PolicyNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim,hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim,hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound
        
    def mu(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.fc_mu(x)
        return mu
   
    def std(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        std = F.softplus(self.fc_std(x)) + 1e-5
        return std

    def dist(self, state):
        mu = self.mu(state)
        std = self.std(state)
        return Normal(mu, std)

    def forward(self, x, deterministic=False): 
        mu = self.mu(x)
        std = self.std(x)
        dist = Normal(mu, std)
        
        if deterministic:
            normal_sample = mu
        else:
            normal_sample = dist.rsample()  
        
        log_prob = dist.log_prob(normal_sample)
        action= torch.tanh(normal_sample) 
        log_prob = log_prob - (2. * (math.log(2.) - normal_sample - F.softplus(-2. * normal_sample)))
        action = action * self.action_bound
        action = torch.clamp(action, -self.action_bound, self.action_bound)
        return action, log_prob
class QValueNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)
class SACContinuous:
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound,
                 actor_lr, critic_lr, alpha_lr, target_entropy, tau, gamma,
                 device, segments, replica):
        self.actor = PolicyNetContinuous(state_dim, hidden_dim, action_dim,
                                         action_bound).to(device)  
        self.critic_1 = QValueNetContinuous(state_dim, hidden_dim,
                                            action_dim).to(device)  
        self.critic_2 = QValueNetContinuous(state_dim, hidden_dim,
                                            action_dim).to(device)  
        self.target_critic_1 = QValueNetContinuous(state_dim,
                                                   hidden_dim, action_dim).to(
                                                       device)  
        self.target_critic_2 = QValueNetContinuous(state_dim,
                                                   hidden_dim, action_dim).to(
                                                       device)  
        self.replay_buffer = ReplayBuffer(100000)
            
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(),
                                                   lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(),
                                                   lr=critic_lr)

        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_alpha.requires_grad = True  
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=alpha_lr)
        self.target_entropy = target_entropy  
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.segments = segments
        self.replica = replica
        
        self.data = [] 
        self.actor_param = []
        self.critic_1_param = []
        self.critic_2_param = []
        self.beta = torch.tensor(0, dtype=torch.float).to(self.device)
        self.policy_advantage = torch.tensor(0, dtype=torch.float).to(self.device)


    def get_param(self):
        self.actor_param = [x.data for x in self.actor.parameters()]


    def take_action(self, state, deterministic):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        action, log_prob = self.actor(state, deterministic)
        return action.item(), log_prob.item() 

    
    def calc_target(self, rewards, next_states, dones):  
        with torch.no_grad():
            next_actions, log_prob = self.actor(next_states) 
            td_target = rewards + self.gamma * (torch.min(
            self.target_critic_1(next_states, next_actions),
            self.target_critic_2(next_states, next_actions)
        ) + self.log_alpha.exp() * (-log_prob)) * (1 - dones)
        return td_target
    

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) +
                                    param.data * self.tau)


    def get_policy_adv(self, model, batch_size, gamma):
        b_s, b_a, b_r, b_ns, b_d, b_a_log  = self.replay_buffer.sample(batch_size)
        transition_dict = {'states': b_s, 'actions': b_a, 'actions_log':b_a_log}
        
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        actions_log = torch.tensor(transition_dict['actions_log'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        
        with torch.no_grad():
            dist_tilde = model.actor.dist(states) 
            log_prob_tilde = dist_tilde.log_prob(actions)
            log_prob_tilde = torch.clamp(log_prob_tilde, -20, 0.0)
            pi_tilde_prob = torch.exp(log_prob_tilde)
            tilde_entropy = dist_tilde.entropy()

            dist_self = self.actor.dist(states) 
            log_prob_self = dist_self.log_prob(actions) 
            log_prob_self = torch.clamp(log_prob_self, -20.0, 0.0)
            pi_self_prob = torch.exp(log_prob_self)
            self_entropy = dist_self.entropy()
            
            q1_value = self.critic_1(states, actions)
            q2_value = self.critic_2(states, actions)
            Q_value = torch.min(q1_value, q2_value)

            off_actions_prob = torch.exp(actions_log)

            entropy_minus = tilde_entropy - self_entropy

            policy_adv = ((pi_tilde_prob - pi_self_prob) / (off_actions_prob + 1e-5)) * Q_value + self.log_alpha.exp() * entropy_minus

        self.policy_advantage = torch.mean(policy_adv)
        Coff = (2 * torch.max(policy_adv) * gamma) / ((1 - gamma) ** 2)
        self.beta = self.get_metric(model, states, actions, Coff)


    def get_metric(self, model, states, actions, Coff):
        appr = True 
        self.get_param()
        model.get_param()
        param_div = [x.data - k.data for k, x in zip(self.actor_param, model.actor_param)]
        param_div_vector = torch.cat([x.view(-1) for x in param_div]).unsqueeze(-1)
        dist_self = self.actor.dist(states) 
        log_prob_self = dist_self.log_prob(actions)
        log_prob_self = torch.clamp(log_prob_self, -20.0, 0.0)
        prob = torch.exp(log_prob_self)
        if appr:
            actor_loss = -log_prob_self.mean()         
            saved_grads = [param.grad.clone() if param.grad is not None else None for param in self.actor.parameters()]
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_grad = [param.grad.data.clone() for param in self.actor.parameters()]
            grad_vector = torch.cat([grad.view(-1) for grad in actor_grad]).unsqueeze(-1)
            FIM = grad_vector @ grad_vector.t()
            beta = torch.sqrt((2 * (self.policy_advantage) / (Coff * (param_div_vector.t() @ FIM @ param_div_vector)))).squeeze()

            for param, saved_grad in zip(self.actor.parameters(), saved_grads):
                if saved_grad is not None:
                    param.grad = saved_grad
        else:
            FIM = None
            batch = states.size(0)
            for i in range(batch):
                self.actor_optimizer.zero_grad()
                actor_loss = -log_prob_self[i]
                actor_loss.backward(retain_graph=True)

                fp16_params = []
                for param in self.actor.parameters():
                    if param.grad is not None:
                        grad_float16 = param.grad.view(-1).detach().to(dtype=torch.float16)
                        param.grad = None 
                        fp16_params.append(grad_float16)
                sample_grad = torch.cat(fp16_params).unsqueeze(-1)
                torch.cuda.empty_cache()
                outer_product = prob[i].to(dtype=torch.float16) * sample_grad @ sample_grad.t()
                if FIM is None:
                    FIM = outer_product
                else:
                    FIM.add_(outer_product)
                del outer_product
                gc.collect()
                torch.cuda.empty_cache()
            del actor_loss
            FIM /= batch
            beta = torch.sqrt((2 * self.policy_advantage.to(dtype=torch.float16) / (Coff.to(dtype=torch.float16) * (param_div_vector.t() @ FIM @ param_div_vector)))).squeeze()
            beta = beta.clone().detach()
            beta = beta.to(dtype=torch.float32)

        return beta
              

    def mix_policy(self, actor_param):
        for k, x in zip(actor_param, self.actor.parameters()):
            x.data.copy_((1 - self.beta) * x.data + self.beta * k.data)


    def get_reshaped_param(self):
        actor_param = [np.array(x.data.cpu()) for x in self.actor.parameters()] 
        return np.array(actor_param, dtype=object)


    def get_segments(self, target_model_weights, p, segments):
        flat_m = []
        shape_list = []
        for x in target_model_weights:
            shape_list.append(x.shape) 
            flat_m.extend(list(x.flatten()))
        seg_length = len(flat_m) // segments + 1 
        return flat_m[p*seg_length:(p+1)*seg_length], shape_list 


    def reconstruct(self,flat_m,shape_list):
        result = []
        current_pos = 0
        for shape in shape_list:
            total_number = 1
            for i in shape:
                total_number *= i 
            result.append(np.array(flat_m[current_pos:current_pos+total_number]).reshape(shape))
            current_pos += total_number
        return np.array(result, dtype=object)
    

    def cache_param(self, actor_param):
        for k, x in zip(actor_param, self.actor.parameters()):
            x.data.copy_(k.data)


    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'],
                               dtype=torch.float).view(-1, 1).to(self.device)

        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)


        td_target = self.calc_target(rewards, next_states, dones)
        critic_1_loss = torch.mean(
            F.mse_loss(self.critic_1(states, actions), td_target.detach()))
        critic_2_loss = torch.mean(
            F.mse_loss(self.critic_2(states, actions), td_target.detach()))
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        clip_grad_norm(self.critic_1.parameters(), 0.5)
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        clip_grad_norm(self.critic_2.parameters(), 0.5)
        self.critic_2_optimizer.step()


        new_actions, log_prob = self.actor(states)
        entropy = -log_prob
        q1_value = self.critic_1(states, new_actions)
        q2_value = self.critic_2(states, new_actions)
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy -
                                torch.min(q1_value, q2_value))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        clip_grad_norm(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()


        alpha_loss = torch.mean(
            (entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)
