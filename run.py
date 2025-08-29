import os
import ssl
import torch
import torch.multiprocessing as mp
from control import BasicMac,NoSharedMac
from env_runner import EpisodeRunner,MultiAgentEnv
from episode_replaybuffer import ReplayBuffer
from utils import set_random_seed,OneHot
from Learner.PPO_learner import PPO_Learner
from Learner.Q_learner import Q_Learner
from Learner.pareto_learner import Pareto_Learner
from config import fetch_args
def run_train(args):
    #检查是否存在args.logs_dir文件夹，如果不存在则创建
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    #检查checkpoint下是否存在args.save_model_dir文件夹，如果不存在则创建
    if not os.path.exists(os.path.join(args.checkpoint_dir, args.save_model_dir)):
        os.makedirs(os.path.join(args.checkpoint_dir, args.save_model_dir))
    #记录下保存模型的路径
    save_model_dir = os.path.join(args.checkpoint_dir, args.save_model_dir)

    #创建环境
    env = MultiAgentEnv(args)
    #创建方案
    scheme = {"state": {"vshape": args.state_dim}, 
              "obs": {"vshape": args.obs_dim, "group": "agents"},
              "actions": {"vshape": (1,), "group": "agents","dtype":torch.long},
              "reward": {"vshape": (args.n_agents, 1)},
              "terminated": {"vshape": (1,),"dtype":torch.uint8}}
    groups = {"agents": args.n_agents}
    preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])}
    
    # 加载ReplayBuffer
    # buffer = ReplayBuffer(scheme, groups, args.buffer_size, args.episode_limit + 1, preprocess=preprocess, device="cpu")
    if args.load_replay_buffer:
        buffer = ReplayBuffer.load("replay_buffer_new.pt", scheme, groups, args.episode_limit + 1, preprocess=preprocess,device="cpu")
    else :
        buffer = ReplayBuffer(scheme, groups, args.buffer_size, args.episode_limit + 1, preprocess=preprocess,device="cpu")

    #Setup multiagent controller here
    # mac = BasicMac(args,scheme)
    mac = NoSharedMac(args,scheme)

    #Setup runner here
    runner = EpisodeRunner(env)
    runner.setup(scheme, groups, preprocess, mac)

    #Learner
    # Learner = Q_Learner(mac=mac,args=args)
    # Learner = PPO_Learner(mac=mac,args=args,scheme=scheme)
    Learner = Pareto_Learner(mac=mac,args=args,scheme=scheme)
    # #导入模型
    # Learner.load_models(save_model_dir)
    if args.use_cuda:
        Learner.cuda()
    
    #Run training
    episode = 0
    while episode < 1000:
        # episode_batch = runner.run(test_mode=False)
        # return 0
        # buffer.insert_episode_batch(episode_batch)
        # # buffer.save("replay_buffer_new819.pt")
        if buffer.can_sample(args.batch_size):           
            batch_sampled = buffer.sample(args.batch_size)

            # Truncate batch to only filled timesteps
            max_ep_t = batch_sampled.max_t_filled()
            batch_sampled = batch_sampled[:, :max_ep_t]

            if batch_sampled.device != 'cuda':
                batch_sampled.to('cuda')
            
            Learner.train(batch_sampled)
            Learner.save_models(save_model_dir)
            episode += 1
        else:
            continue
    # buffer.save("replay_buffer.pt")
    #保存模型
    # Learner.save_models(save_model_dir)
    print("Training complete")

if __name__ == "__main__":
    # Change working directory to script's directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    ssl._create_default_https_context = ssl._create_unverified_context
    mp.set_start_method('spawn')
    checkpoint_dir = './checkpoint/'
    figure_dir = './figures/'
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)
    set_random_seed(127)
    args = fetch_args()
    run_train(args)