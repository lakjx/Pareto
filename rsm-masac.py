import torch
import os
import numpy as np
import setproctitle
from tqdm import tqdm
import pickle

from types import SimpleNamespace

from env_runner import MultiAgentEnv 
from rsm_masac_components import SACDiscrete 

class DummyEnv:
    """
    一个快速的虚拟多智能体环境，用于调试和测试训练流程。
    它完美模仿了 MultiAgentEnv 的接口，但内部完全由随机数驱动。
    """
    def __init__(self, args):
        print("="*50)
        print("INFO: Using a fast Dummy Environment for testing.")
        print("="*50)
        
        # 1. 复制所有必要的属性，确保与真实环境一致
        self.num_agents = args.n_agents
        self.action_space_size = args.n_actions
        self.observation_space_dim = args.obs_dim
        self.state_space_dim = args.state_dim
        self.episode_limit = args.episode_limit
        self.args = args # 保存args以备他用

        # 内部状态
        self._current_step = 0
        self.obs = None
        self.state = None
        
        # 2. 模仿真实环境中被调用的 FL_env 属性
        # 创建一个模拟对象，它有一个什么都不做的 save_metrics_to_excel 方法
        class MockFLServer:
            def save_metrics_to_excel(self, save_dir):
                # 这个方法在测试时会被调用，但我们让它什么都不做
                # print(f"MockFLServer: Pretending to save metrics to {save_dir}")
                pass
        
        # 创建一个包含模拟对象的列表
        self.FL_env = [MockFLServer() for _ in range(self.num_agents)]

    def reset(self):
        """重置环境，返回一个随机的初始观测。"""
        self._current_step = 0
        # 生成随机的初始局部观测
        self.obs = np.random.randn(self.num_agents, self.observation_space_dim).astype(np.float32)
        # 生成随机的初始全局状态
        self.state = np.random.randn(self.state_space_dim).astype(np.float32)
        return self.obs

    def step(self, actions):
        """
        执行一个时间步，返回随机的奖励、完成状态和信息。
        `actions` 的内容在这里被忽略，因为逻辑是随机的。
        """
        self._current_step += 1

        # 1. 生成随机的奖励，形状为 (num_agents, 1)，与真实环境匹配
        rewards = np.random.randn(self.num_agents, 1).astype(np.float32)

        # 2. 判断是否结束
        done = self._current_step >= self.episode_limit

        # 3. 生成随机的下一个状态和观测
        self.obs = np.random.randn(self.num_agents, self.observation_space_dim).astype(np.float32)
        self.state = np.random.randn(self.state_space_dim).astype(np.float32)
        
        # 4. 模仿真实环境返回一个虚拟的 info
        dummy_info = [0] * self.num_agents

        return rewards, done, dummy_info

    def get_obs(self):
        """返回当前的局部观测。"""
        return self.obs

    def get_state(self):
        """返回当前的全局状态。"""
        return self.state

# 适配后的 RSM-MASAC 控制器类
# ##############################################################################

class RSM_MASAC:
    def __init__(self, env, args):
        self.env = env
        self.args = args
        self.num_agents = self.env.num_agents
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Initializing RSM-MASAC for discrete action space of size {args.n_actions}.")

        # 核心改动：为每个智能体创建SAC实例，并进行适配
        # 注意：SAC的state_dim是单个智能体的obs_dim
        # PolicyNetContinuous输出的action_dim为1（一个连续值），action_bound为1.0
        self.agents = [
            SACDiscrete(
                state_dim=args.obs_dim,
                hidden_dim=args.hidden_dim,
                action_dim=args.n_actions, # 输出一个1维连续动作
                # action_bound=1.0, # 动作范围为[-1, 1]
                actor_lr=args.actor_lr,
                critic_lr=args.critic_lr,
                alpha_lr=args.alpha_lr,
                target_entropy=-1.0, # -action_dim
                tau=args.tau,
                gamma=args.gamma,
                device=self.device,
                segments=args.segments,
                replica=args.replica
            ) for _ in range(self.num_agents)
        ]
        
        # 缓存智能体，用于RSM机制
        self.cache_agents = [
            SACDiscrete(
                state_dim=args.obs_dim, hidden_dim=args.hidden_dim, action_dim =args.n_actions,
                actor_lr=args.actor_lr, critic_lr=args.critic_lr,
                alpha_lr=args.alpha_lr, target_entropy=-1.0, tau=args.tau,
                gamma=args.gamma, device=self.device, segments=args.segments, replica=args.replica
            ) for _ in range(self.num_agents)
        ]
        self.episode = 0

    def get_actions(self, obs, deterministic=False):
        """
        为所有智能体生成动作。
        这是适配的核心，将连续动作转换为离散动作。
        """
        log_probs = []
        actions = []
        obs_tensor = torch.tensor(obs, dtype=torch.float).to(self.device)
        for i in range(self.num_agents):
            agent_obs = obs_tensor[i]
            action, log_prob = self.agents[i].take_action(agent_obs, deterministic)
            actions.append(action)
            log_probs.append(log_prob)
        return np.array(actions), np.array(log_probs)

    def _perform_rsm_step(self):
        for i in range(self.num_agents):
            agent_i = self.agents[i]
            potential_neighbors = [j for j in range(self.num_agents) if j != i]
            if not potential_neighbors: continue

            segments_ = min(agent_i.segments, len(potential_neighbors))
            replica_ = min(agent_i.replica, len(potential_neighbors))

            for _ in range(replica_):
                np.random.shuffle(potential_neighbors)
                target_agents_indices = potential_neighbors[:segments_]
                
                actor_reconstruct_list = []
                shape_record = None
                for p, target_idx in enumerate(target_agents_indices):
                    target_actor_weights = self.agents[target_idx].get_reshaped_param()
                    seg, shape_record = agent_i.get_segments(target_actor_weights, p, segments_)
                    # actor_reconstruct_list.extend(seg)
                    actor_reconstruct_list.append(seg)
                
                if shape_record is None: continue
                avg_actor_sum =agent_i.reconstruct(actor_reconstruct_list, shape_record)
                
                # torch_params = [torch.from_numpy(p).float().to(self.device) for p in avg_actor_sum]
                
                self.cache_agents[i].cache_param(avg_actor_sum)
                if self.agents[i].replay_buffer.size() > self.args.batch_size:
                    self.agents[i].get_policy_adv(self.cache_agents[i], self.args.batch_size, self.args.gamma)
                    policy_adv_judge = agent_i.policy_advantage.cpu().detach().numpy().item()
                    agent_i.beta = torch.clamp(agent_i.beta, 0.0, 0.8)
                    if policy_adv_judge > 0.0:
                        agent_i.mix_policy(avg_actor_sum)

    def train(self):
        """
        在线训练主循环
        """
        print("Starting online training for RSM-MASAC...")
        for episode in range(self.args.num_episodes):
            self.episode = episode
            obs = self.env.reset()
            done = False
            episode_rewards = np.zeros(self.num_agents)
            
            for t in tqdm(range(self.args.episode_limit), desc=f"Episode {episode}"):
                actions, log_probs = self.get_actions(obs)

                # 2. 使用离散动作与环境交互
                rewards, done, _ = self.env.step(actions)
                rewards = rewards.squeeze() # 确保rewards是一维数组
                
                next_obs = self.env.get_obs()
                
                # 3. 存储经验时，使用连续动作
                for i in range(self.num_agents):
                    self.agents[i].replay_buffer.add(
                        obs[i], actions[i], rewards[i], next_obs[i], done, log_probs[i]
                    )

                obs = next_obs
                episode_rewards += rewards
                
                # 4. 从回放池采样并更新网络
                if self.agents[0].replay_buffer.size() > self.args.batch_size:
                    for agent in self.agents:
                        b_s, b_a, b_r, b_ns, b_d, _ = agent.replay_buffer.sample(self.args.batch_size)
                        transition_dict = {
                            'states': b_s, 'actions': b_a, 'next_states': b_ns,
                            'rewards': b_r, 'dones': b_d
                        }
                        agent.update(transition_dict)
                
                # 5. 周期性执行RSM策略混合
                if t > 0 and t % self.args.comm_interval == 0:
                    self._perform_rsm_step()
                
                if done:
                    break

            print(f"Episode {episode} finished. Total Rewards: {episode_rewards.sum():.2f}")

            if (episode + 1) % 50 == 0:
                self.save_model(self.args.model_save_dir)
    def load_buffer(self, buffer_files):
        """
        核心数据桥梁函数，与之前讨论的 _prepare_data_from_mappo_buffer 功能相同。
        它负责加载MAPPO格式的buffer文件，并将其中的轨迹数据转换为逐条的转移数据，
        填充到每个SAC智能体的ReplayBuffer中。
        """
        print(f"Loading data from MAPPO buffer files: {buffer_files}...")
        
        # 1. 从多个pickle文件中加载所有经验
        mappo_experiences = []
        for filename in buffer_files:
            if not os.path.exists(filename):
                print(f"Warning: Buffer file not found at {filename}")
                continue
            with open(filename, 'rb') as f:
                mappo_experiences.extend(pickle.load(f))
        
        if not mappo_experiences:
            raise ValueError("No experiences loaded. Please check buffer file paths.")

        # 2. 将加载的列表数据转换回Numpy数组
        for i in range(len(mappo_experiences)):
            for k, v in mappo_experiences[i].items():
                 mappo_experiences[i][k] = np.array(v)

        # 3. 遍历所有经验，构建 (s, a, r, s', d) 转移并存入回放池
        total_transitions = 0
        print("Converting trajectories to transitions for replay buffers...")
        for idx in range(len(mappo_experiences) - 1): # 遍历到倒数第二个，以方便获取next_obs
            exp = mappo_experiences[idx]
            next_exp = mappo_experiences[idx+1]
            
            # 如果当前时间步是某个episode的终点，那么它的下一个状态是无效的
            # 我们应该跳过这个转换，因为它不构成一个有效的(s, a, r, s')
            if exp['done']:
                continue

            for agent_i in range(self.num_agents):
                obs = exp['obs'][agent_i]
                # 在离线训练中，我们需要的是SAC智能体当初输出的连续动作
                # 但MAPPO buffer中存的是离散动作。这是一个关键差异。
                # 简化处理：我们先用离散动作填充，但理想情况下，buffer应存连续动作。
                # 此处我们假设 'actions' 字段可以被SAC Critic网络接受。
                # 对于一个更鲁棒的实现，当初收集数据时就应该保存连续动作。
                # 这里我们暂时使用离散动作作为占位符。
                action = exp['actions'][agent_i] 
                reward = exp['rewards'][agent_i]
                next_obs = next_exp['obs'][agent_i]
                done = exp['done']
                action_log = exp['log_probs'][agent_i]
                
                # 将转换后的数据添加到对应智能体的回放池
                # 注意：action应该是连续值，这里暂时用离散值代替
                self.agents[agent_i].replay_buffer.add(obs, action, reward, next_obs, done, action_log)
                total_transitions += 1

        print(f"Data loading complete. Loaded {total_transitions} total transitions.")


    def pure_train(self, buff_dir_list):
        self.load_buffer(buff_dir_list)

        print("\nStarting pure offline training for RSM-MASAC...")
        
        # 检查数据是否足够
        if self.agents[0].replay_buffer.size() < self.args.batch_size:
            print("Buffer size is too small to start training. Please collect more data.")
            return

        # 第二步：进行固定次数的梯度更新
        # 借用MAPPO的参数来确定总更新次数，以进行公平对比
        total_gradient_steps = self.args.num_episodes * self.args.episode_limit 
        
        comm_step_counter = 0
        for step in tqdm(range(total_gradient_steps), desc="Offline Training RSM-MASAC"):
            self.episode = (step + 1) // self.args.episode_limit # 更新episode计数器用于保存模型

            # 为每个智能体执行一次SAC更新
            for agent in self.agents:
                b_s, b_a, b_r, b_ns, b_d, _ = agent.replay_buffer.sample(self.args.batch_size)
                transition_dict = {
                    'states': b_s, 'actions': b_a, 'next_states': b_ns,
                    'rewards': b_r, 'dones': b_d
                }
                agent.update(transition_dict)
            
            # 周期性地执行RSM策略混合
            if (step + 1) % self.args.comm_interval == 0:
                self._perform_rsm_step() # 此方法已在上一版代码中提供
                comm_step_counter += 1

            # 定期保存模型
            if (step + 1) % (2) == 0: 
                print(f"\nStep {step+1}/{total_gradient_steps}, saving model...")
                self.save_model(self.args.model_save_dir) # 此方法已在上一版代码中提供

        print("Pure offline training finished.")
        self.save_model(self.args.model_save_dir)

    def save_model(self, path):
        path_with_episode = path
        os.makedirs(path_with_episode, exist_ok=True)
        for i, agent in enumerate(self.agents):
            agent_path = os.path.join(path_with_episode, f'agent_{i}.pth')
            torch.save({
                'actor_state_dict': agent.actor.state_dict(),
                'critic_1_state_dict': agent.critic_1.state_dict(),
                'critic_2_state_dict': agent.critic_2.state_dict(),
            }, agent_path)
        print(f"Saved models for {self.num_agents} agents at {path_with_episode}")
    
    def load_model(self, path):
        """
        加载所有智能体的模型参数。
        这个方法会从指定目录中，为每个智能体加载对应的模型文件(agent_0.pth, agent_1.pth, ...)。
        """
        print(f"Loading models from {path}...")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model directory not found at {path}")

        for i, agent in enumerate(self.agents):
            agent_path = os.path.join(path, f'agent_{i}.pth')
            if not os.path.exists(agent_path):
                raise FileNotFoundError(f"Model file for agent {i} not found at {agent_path}")

            # 加载检查点文件
            checkpoint = torch.load(agent_path, map_location=self.device)
            
            # 加载Actor和两个Critic网络的参数
            agent.actor.load_state_dict(checkpoint['actor_state_dict'])
            agent.critic_1.load_state_dict(checkpoint['critic_1_state_dict'])
            agent.critic_2.load_state_dict(checkpoint['critic_2_state_dict'])
            
            # 关键：加载后必须同步目标网络(Target Networks)的参数
            agent.target_critic_1.load_state_dict(agent.critic_1.state_dict())
            agent.target_critic_2.load_state_dict(agent.critic_2.state_dict())
        
        print(f"Models for {self.num_agents} agents loaded successfully.")


    def test(self, load_dir, sav_dir):
        """
        测试训练好的模型。
        该方法的结构与您提供的MAPPO.test方法完全一致。
        """
        self.load_model(load_dir)
        
        obs = self.env.reset()
        done = False
        t = 0
        
        print("Starting testing...")
        
        while not done:
            discrete_actions,_ = self.get_actions(obs, deterministic=True)

            rewards, done, _ = self.env.step(discrete_actions)
            
            done = True if t >= self.env.episode_limit - 1 else done
            t += 1
            
            next_obs = self.env.get_obs()
            
            obs = next_obs
        
        print("Testing finished. Saving metrics...")
        
        for id_ in range(self.num_agents):
            # 假设您的env对象中有FL_env属性且实现了save_metrics_to_excel方法
            self.env.FL_env[id_].save_metrics_to_excel(sav_dir)
        
        print(f"Metrics saved to {sav_dir}")

# ==============================================================================
# 适配后的参数配置与主函数
# ==============================================================================

def get_args():
    tasks = ['MNIST', 'FashionMNIST', 'CIFAR10']
    env_name = f'a{len(tasks)}'
    
    args_dict = {
        'alg': 'rsm-masac',
        'n_agents': len(tasks),
        'obs_dim': 9 + len(tasks),
        'state_dim': 2 + 4 * len(tasks),
        'n_actions': 27, # 离散动作空间大小
        'action_is_mix': False, # 确保使用您环境的离散动作逻辑
        'use_direct_actions': False, # 同上，使用您的增量式动作
        
        # RSM-MASAC Params
        'hidden_dim': 256,
        'alpha_lr': 3e-4,
        'tau': 0.005,
        'segments': 4,
        'replica': 3,
        'comm_interval': 3, 

        # Shared Params
        'actor_lr': 1e-4,
        'critic_lr': 3e-4,
        'gamma': 0.99,
        'batch_size': 256,
        'num_episodes': 1000,
        'episode_limit': 35,
        
        # Env Params
        'dataset_names': tasks,
        'n_clients': 5,
        'non_iid_level': 1,
        
        # Path Params
        'model_save_dir': f'./checkpoint/rsm_masac_{env_name}',
        'env_name': env_name,
    }
    return SimpleNamespace(**args_dict)

if __name__ == '__main__':
    is_testing = True 
    args = get_args()
    torch.manual_seed(100)
    np.random.seed(100)
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    setproctitle.setproctitle(f'rsm_masac_{args.env_name}')
    
    # 初始化环境
    env = MultiAgentEnv(args)
    # env = DummyEnv(args)  # 使用快速的虚拟环境进行测试
    
    # 初始化适配后的RSM-MASAC控制器
    agent_controller = RSM_MASAC(env, args)
    mappo_buffer_files = [
        'tmc_last/mappobuffer_324.pkl' 
    ]

    

    if is_testing:
        # 定义模型加载路径和结果保存路径
        model_load_directory = 'pareto_exp/checkpoint/rsm_masac_a3/' # 示例路径
        results_save_directory = 'pareto_exp/results/rsm_masac_a3'
        os.makedirs(results_save_directory, exist_ok=True)
        
        agent_controller.test(model_load_directory, results_save_directory)
    else:
        # 运行训练流程
        agent_controller.train()
        # agent_controller.pure_train(mappo_buffer_files)