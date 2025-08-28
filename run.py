import os
import ssl
import torch
import setproctitle
import torch.multiprocessing as mp
import time
from control import BasicMac,NoSharedMac
from env_runner import EpisodeRunner,MultiAgentEnv
from episode_replaybuffer import ReplayBuffer
from utils import set_random_seed,OneHot
from Learner.PPO_learner import PPO_Learner
from Learner.Q_learner import Q_Learner
from Learner.pareto_learner import Pareto_Learner
from config import fetch_args
def run_train(args):
    setproctitle.setproctitle(args.exp_name)
    args.is_test = args.is_test == 1
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
        buffer = ReplayBuffer.load(args.replay_buffer_root, scheme, groups, args.episode_limit + 1, preprocess=preprocess,device="cpu")
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
    if args.is_test:
        Learner.load_models(save_model_dir)
    if args.use_cuda:
        Learner.cuda()
    # print_model_flops(Learner)
    #Run training
    episode = 0
    while episode < 1000:
        if args.is_test:
            runner.run(test_mode=True,excel_dir=args.excel_dir) 
            return 0
        # episode_batch = runner.run()
        # buffer.insert_episode_batch(episode_batch)
        # if episode % 50 == 0 and episode > 0:
        #     buffer.save(args.replay_buffer_root)
        if buffer.can_sample(args.batch_size):           
            batch_sampled = buffer.sample(args.batch_size)

            # Truncate batch to only filled timesteps
            max_ep_t = batch_sampled.max_t_filled()
            batch_sampled = batch_sampled[:, :max_ep_t]

            if batch_sampled.device != 'cuda':
                batch_sampled.to('cuda')
            t0 = time.time()
            Learner.train(batch_sampled)
            Learner.save_models(save_model_dir)
            print(f"Episode training time: {time.time() - t0}s")
            episode += 1
        else:
            continue
    # buffer.save("replay_buffer.pt")
    #保存模型
    # Learner.save_models(save_model_dir)
    print("Training complete")

def print_model_flops(learner):
    flops_info = learner.get_flops()
    
    print(f"=== 模型FLOPs分析 ===")
    print(f"总FLOPs: {flops_info['total_flops'] / 1e6:.2f} MFLOPs")
    print(f"总参数量: {flops_info['total_params'] / 1e3:.2f} K")
    
    print(f"\n=== 组件详情 ===")
    print(f"Critic网络: {flops_info['critic_flops'] / 1e9:.2f} GFLOPs, {flops_info['critic_params'] / 1e6:.2f} M参数")
    print(f"State Value网络: {flops_info['state_value_flops'] / 1e9:.2f} GFLOPs, {flops_info['state_value_params'] / 1e6:.2f} M参数")
    print(f"对手动作预测器: {flops_info['oppo_pred_flops'] / 1e9:.2f} GFLOPs, {flops_info['oppo_pred_params'] / 1e6:.2f} M参数")
    print(f"MAC网络: {flops_info['mac_flops'] / 1e9:.2f} GFLOPs, {flops_info['mac_params'] / 1e6:.2f} M参数")

if __name__ == "__main__":
    # Change working directory to script's directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    ssl._create_default_https_context = ssl._create_unverified_context
    mp.set_start_method('spawn')
    set_random_seed(1611)
    args = fetch_args()
    
    run_train(args)