import torch
import os
import pickle
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from collections import deque
from types import SimpleNamespace
from env_runner import MultiAgentEnv
class MultiAgentActor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.shared_net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.action_head = nn.Linear(64, action_dim)
    
    def forward(self, obs):
        # obs形状: (batch_size, num_agents, obs_dim) 或 (num_agents, obs_dim)
        original_shape = obs.shape
        if len(original_shape) > 2:
            obs = obs.view(-1, original_shape[-1])  # 展平前两维
        
        features = self.shared_net(obs)
        logits = self.action_head(features)
        
        if len(original_shape) > 2:
            logits = logits.view(*original_shape[:-1], -1)  # 恢复原始形状
        return logits

class CentralizedCritic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, states):
        # states形状: (batch_size, state_dim)
        return self.net(states)

class MAPPO:
    def __init__(self, env, args):
        self.env = env
        self.args = args
        self.num_agents = self.env.num_agents

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 初始化多智能体策略网络
        self.actor = MultiAgentActor(args.obs_dim, args.n_actions).to(self.device)
        self.critic = CentralizedCritic(args.state_dim).to(self.device)
        
        # 优化器
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=args.critic_lr)
        
        # 经验缓冲区
        self.buffer = deque(maxlen=args.buffer_size)
        
        # 超参数
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self.ppo_clip = args.clip_epsilon
        self.epochs = args.ppo_epochs

        self.writer = SummaryWriter(f'logs/mappo/{args.env_name}')

    def get_actions(self, obs,deterministic=False):
        """为所有智能体生成动作"""
        # obs形状: (num_agents, obs_dim)
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).to(self.device)
            logits = self.actor(obs_tensor)  # (num_agents, action_dim)
        
        dists = [torch.distributions.Categorical(logits=logit) for logit in logits]
        if deterministic:
            actions = [dist.probs.argmax().item() for dist in dists]
        else:
            actions = [dist.sample() for dist in dists]
        log_probs = torch.stack([dists[i].log_prob(actions[i]) for i in range(self.num_agents)])
        
        return (torch.stack(actions).cpu().numpy(), 
                log_probs.cpu().numpy())

    def store_experience(self, experience):
        """存储所有智能体的联合经验"""
        self.buffer.append(experience)

    def compute_gae(self, rewards, values, next_values, dones):
        """计算GAE优势函数"""
        # 确保values和rewards形状一致
        values = values.reshape(-1, 1)  # (buffer_length, 1)
        next_values = next_values.reshape(-1, 1)  # (buffer_length, 1)
        dones = dones.reshape(-1, 1)  # (buffer_length, 1)

        advantages = np.zeros_like(rewards)
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_values[t]
            else:
                next_value = values[t+1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_advantage
            last_advantage = advantages[t]
        
        returns = advantages + values
        return advantages, returns

    def update(self):
        # 从缓冲区获取数据
        batch = list(self.buffer)
        obs = np.stack([exp['obs'] for exp in batch])          # (batch_size, num_agents, obs_dim)
        states = np.stack([exp['state'] for exp in batch])      # (batch_size, state_dim)
        actions = np.stack([exp['actions'] for exp in batch])   # (batch_size, num_agents)
        old_log_probs = np.stack([exp['log_probs'] for exp in batch]) # (batch_size, num_agents)
        rewards = np.stack([exp['rewards'] for exp in batch])   # (batch_size, num_agents)
        next_states = np.stack([exp['next_state'] for exp in batch]) # (batch_size, state_dim)
        dones = np.array([exp['done'] for exp in batch])        # (batch_size,)
        
        # 计算当前状态值和下一个状态值
        states_tensor = torch.FloatTensor(states).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)        
        current_values = self.critic(states_tensor).squeeze().cpu().detach().numpy()
        next_values = self.critic(next_states_tensor).squeeze().cpu().detach().numpy()
        
        # 计算每个智能体的GAE和returns
        all_advantages = []
        all_returns = []
        for i in range(self.num_agents):
            agent_rewards = rewards[:, i].reshape(-1, 1)
            advantages, returns = self.compute_gae(agent_rewards, current_values, next_values, dones)
            all_advantages.append(advantages)
            all_returns.append(returns)
        
        # 转换为张量
        obs_tensor = torch.FloatTensor(obs).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(old_log_probs).to(self.device)
        advantages_tensor = torch.FloatTensor(np.array(all_advantages).transpose(1, 0, 2).squeeze(-1)).to(self.device)  # (batch_size, num_agents)
        returns_tensor = torch.FloatTensor(np.array(all_returns).transpose(1, 0, 2).squeeze(-1)).to(self.device)  # (batch_size, num_agents)

        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

        epoch_actor_loss = 0
        epoch_critic_loss = 0
        # PPO更新
        for _ in range(self.epochs):
            # 随机打乱数据
            indices = np.arange(len(batch))
            np.random.shuffle(indices)

            for start in range(0, len(indices), self.args.batch_size):
                end = start + self.args.batch_size
                batch_indices = indices[start:end]
                
                # 获取当前批数据
                batch_obs = obs_tensor[batch_indices]          # (batch_size, num_agents, obs_dim)
                batch_actions = actions_tensor[batch_indices]  # (batch_size, num_agents)
                batch_old_log_probs = old_log_probs_tensor[batch_indices] # (batch_size, num_agents)
                batch_advantages = advantages_tensor[batch_indices] # (batch_size, num_agents)
                batch_returns = returns_tensor[batch_indices] # (batch_size, num_agents)
                batch_states = states_tensor[batch_indices]

                # 计算新策略的概率
                logits = self.actor(batch_obs)  # (batch_size, num_agents, action_dim)
                new_log_probs = torch.stack([
                    torch.distributions.Categorical(logits=logits[:,i]).log_prob(batch_actions[:,i]) 
                    for i in range(self.num_agents)
                ], dim=1)  # (batch_size, num_agents)
                
                # 计算ratio
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # 计算Actor损失
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1-self.ppo_clip, 1+self.ppo_clip) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # 计算Critic损失
                current_values = self.critic(batch_states).squeeze()
                critic_loss = nn.MSELoss()(current_values, batch_returns.mean(dim=1))  # 使用平均回报
                
                # 参数更新
                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.actor_optim.step()
                
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

                epoch_actor_loss += actor_loss.item()
                epoch_critic_loss += critic_loss.item()

        # 记录训练信息
        self.writer.add_scalar('actor_loss', epoch_actor_loss, self.episode)
        self.writer.add_scalar('critic_loss', epoch_critic_loss, self.episode)
        # self.buffer.clear() if len(self.buffer) >= self.args.buffer_size else None

    def train(self):
        for episode in range(self.args.num_episodes):
            print(f'----------------Episode {episode}---------------------')
            self.episode = episode
            obs = self.env.reset()
            state = self.env.get_state()
            done = False
            episode_rewards = np.zeros(self.env.num_agents) 
            t = 0
            while not done:
                # 获取所有智能体的动作
                actions, log_probs = self.get_actions(obs)

                # 与环境交互
                rewards, done, _ = self.env.step(actions)
                done = True if t >= self.env.episode_limit else False
                t += 1
                next_obs = self.env.get_obs()
                next_state = self.env.get_state()
                
                # 存储联合经验
                experience = {
                    'obs': obs.copy(),
                    'state': state.copy(),
                    'actions': actions,
                    'log_probs': log_probs,
                    'rewards': rewards.squeeze(),  # (num_agents,)
                    'next_state': next_state.copy(),
                    'done': done
                }
                
                self.store_experience(experience)
                
                # 更新状态
                obs = next_obs
                state = next_state
                episode_rewards += rewards.squeeze()

            episode_rewards_str = ' '.join(map(str, episode_rewards))
            print("-------An environment episode is done.------")
            print(f"Episode rewards: {episode_rewards_str}")
            # 记录训练信息
            self.writer.add_scalar(f'rwd_agent_0', episode_rewards[0], episode)
            self.writer.add_scalar(f'rwd_agent_1', episode_rewards[1], episode)
            self.writer.add_scalar(f'rwd_agent_2', episode_rewards[2], episode)

            # 执行PPO更新
            if len(self.buffer) >= self.args.batch_size:
                print("Start to update the model......")
                self.update()
            if episode % 50 == 0 and episode > 0:
                self.save_buffer(self.args.buffer_save_dir)
            # if episode % 1 == 0 and episode > 0:
                self.save_model(self.args.model_save_dir)
        self.writer.close()
    
    def test(self,load_dir):
        self.load_model(load_dir)
        obs = self.env.reset()
        done = False
        t = 0
        while not done:
            # 获取所有智能体的动作
            actions, log_probs = self.get_actions(obs,deterministic=True)

            # 与环境交互
            rewards, done, _ = self.env.step(actions)
            done = True if t >= self.env.episode_limit else False
            t += 1
            next_obs = self.env.get_obs()
            
            # 更新状态
            obs = next_obs
        for id_ in range(self.num_agents):
            self.env.FL_envs[id_].save_metrics_to_excel('results/mappo/')

    def pure_train(self,buff_dir):
        self.load_buffer(buff_dir)
        for episode in range(self.args.num_episodes):
            print(f'----------------Episode {episode}---------------------')
            self.episode = episode
            
            self.update()
        self.save_model(self.args.model_save_dir)


    def save_model(self,path):
        path_with_episode = os.path.join(path, f'ep_{self.episode}/')
        os.makedirs(path_with_episode, exist_ok=True)
        torch.save(self.actor.state_dict(), path_with_episode+'actor.pth')
        torch.save(self.critic.state_dict(), path_with_episode+'critic.pth')
        torch.save(self.actor_optim.state_dict(), path_with_episode+'actor_optim.pth')
        torch.save(self.critic_optim.state_dict(), path_with_episode+'critic_optim.pth')
    
    def load_model(self,path):
        self.actor.load_state_dict(torch.load(path+'/actor.pth'))
        self.critic.load_state_dict(torch.load(path+'/critic.pth'))
        self.actor_optim.load_state_dict(torch.load(path+'/actor_optim.pth'))
        self.critic_optim.load_state_dict(torch.load(path+'/critic_optim.pth'))
    
    def save_buffer(self,filename):
        filename = filename + '.pkl'
        buffer_list = list(self.buffer)       
        # 转换numpy数组为可序列化格式
        serializable_buffer = []
        for exp in buffer_list:
            serialized_exp = {
                'obs': exp['obs'].tolist(),
                'state': exp['state'].tolist(),
                'actions': exp['actions'].tolist(),
                'log_probs': exp['log_probs'].tolist(),
                'rewards': exp['rewards'].tolist(),
                'next_state': exp['next_state'].tolist(),
                'done': exp['done']
            }
            serializable_buffer.append(serialized_exp)
        
        with open(filename, 'wb') as f:
            pickle.dump(serializable_buffer, f)
        print(f"Saved buffer with {len(buffer_list)} experiences to {filename}")
    def load_buffer(self,filename):
        """从文件加载缓冲区"""
        filename = filename or self.buffer_path
        if not os.path.exists(filename):
            print(f"No buffer file found at {filename}")
            return
        
        with open(filename, 'rb') as f:
            serializable_buffer = pickle.load(f)
        
        # 转换回numpy格式
        self.buffer.clear()
        for exp in serializable_buffer:
            restored_exp = {
                'obs': np.array(exp['obs'], dtype=np.float32),
                'state': np.array(exp['state'], dtype=np.float32),
                'actions': np.array(exp['actions'], dtype=np.int64),
                'log_probs': np.array(exp['log_probs'], dtype=np.float32),
                'rewards': np.array(exp['rewards'], dtype=np.float32),
                'next_state': np.array(exp['next_state'], dtype=np.float32),
                'done': exp['done']
            }
            self.buffer.append(restored_exp)
        print(f"Loaded {len(self.buffer)} experiences from {filename}")


def get_args():
    dict = {
        'n_agents': 5,
        'obs_dim': 14,
        'state_dim': 22,
        'n_actions': 27,
        'actor_lr': 0.001,
        'critic_lr': 0.001,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'ppo_epochs': 10,
        'clip_epsilon': 0.2,
        'batch_size': 64,
        'buffer_size': 4096,
        'num_episodes': 1000,
        'episode_limit': 15,

        'dataset_names': ['MNIST', 'FashionMNIST', 'CIFAR10', 'QMNIST', 'SVHN'],
        'n_clients': 5,
        'non_iid_level': 1,
        'action_is_mix': False,

        'model_save_dir': 'checkpoint/mappo_a5/',
        'buffer_save_dir': 'buffer_mappo_a5',
        'env_name': 'mappo_a5',
    }
    return SimpleNamespace(**dict)


if __name__ == '__main__':
    args = get_args()
    env = MultiAgentEnv(args)
    agent = MAPPO(env, args)
    # agent.pure_train('mappobuffer_debug.pkl')
    agent.train()