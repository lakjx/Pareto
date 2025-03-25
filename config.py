import argparse


def fetch_args():
    tasks = ['MNIST', 'FashionMNIST', 'CIFAR10','QMNIST','SVHN']
    exp_name = f'pac-c_a{len(tasks)}'
    # exp_name = f'pac_a{len(tasks)}'
    # 创建一个解析器
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--exp_name', type=str, default=exp_name, help='exp_name')
    parser.add_argument('--seed', type=int, default=156862, help='seed')
    #FL
    parser.add_argument('--n_clients', type=int, default=5, help='n_clients')
    parser.add_argument('--dataset_names',type=list,default=tasks,help='dataset_names')
    parser.add_argument('--non_iid_level', type=float, default=1, help='non_iid_alpha')
    parser.add_argument('--expectile', type=float, default=0.5, help='expectile')
    #env
    parser.add_argument('--n_agents', type=int, default=len(tasks), help='n_agents')
    parser.add_argument('--n_actions', type=int, default=27, help='n_actions') #27
    parser.add_argument('--obs_dim', type=int, default=9+len(tasks), help='obs_dim')
    parser.add_argument('--state_dim', type=int, default=2+4*len(tasks), help='state_dim')
    parser.add_argument('--action_is_mix', type=bool, default=False, help='action_is_mix')
    parser.add_argument('--episode_limit', type=int, default=15, help='episode_limit')
    parser.add_argument('--buffer_size', type=int, default=1000, help='buffer_size')

    parser.add_argument('--agent_output_type', type=str, default='pi_logits', help='agent_output_type')
    parser.add_argument('--evaluation_epsilon', type=float, default=0.05, help='evaluation_epsilon')
    parser.add_argument('--training_epsilon', type=float, default=0.1, help='training_epsilon')
    parser.add_argument('--obs_last_action', type=bool, default=False, help='obs_last_action')
    parser.add_argument('--obs_agent_id', type=bool, default=True, help='obs_agent_id')
    parser.add_argument('--obs_individual_obs', type=bool, default=False, help='obs_individual_obs')

    parser.add_argument('--add_value_last_step', type=bool, default=True, help='add_value_last_step')
    #PPO
    parser.add_argument('--q_nstep', type=int, default=10, help='q_nstep')
    parser.add_argument('--ppo_epochs', type=int, default=5, help='ppo_epoch')
    parser.add_argument('--ppo_batch_size', type=int, default=32, help='ppo_batch_size')
    parser.add_argument('--ppo_clip_param', type=float, default=0.2, help='ppo_clip_param')
    parser.add_argument('--entropy_coef', type=float, default=0.01, help='entropy_coef')
    parser.add_argument('--target_update_interval_or_tau', type=int, default=50, help='critic_training_steps')

    #MIX
    parser.add_argument('--use_cuda', type=bool, default=True, help='use_cuda')
    parser.add_argument('--mixer', type=str, default='qmix', help='mixer')
    parser.add_argument('--mixing_embed_dim', type=int, default=32, help='mixing_embed_dim')
    parser.add_argument('--hypernet_layers', type=int, default=2, help='hypernet_layers')
    parser.add_argument('--hypernet_embed', type=int, default=64, help='hypernet_embed')
    parser.add_argument('--use_rnn', type=bool, default=False, help='use_rnn')
    parser.add_argument('--hidden_dim', type=int, default=64, help='hidden_dim')
    
    

    #optimizer
    parser.add_argument('--lr', type=float, default=1e-4, help='lr') #5e-3
    parser.add_argument('--optim_weight_decay', type=float, default=0.0001, help='optim_weight_decay')

    parser.add_argument('--standardise_rewards', type=bool, default=True, help='standardise_rewards')

    parser.add_argument('--doubleQ', type=bool, default=True, help='doubleQ')

    #RL
    parser.add_argument('--episode_max_steps', type=int, default=100, help='episode_max_steps')
    parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
    parser.add_argument('--gamma', type=float, default=0.6, help='gamma')
    parser.add_argument('--nstep_return', type=int, default=1, help='nstep_return')
    parser.add_argument('--grad_norm_clip', type=float, default=10, help='grad_norm_clip')
    parser.add_argument('--targetnets_updatefreq', type=int, default=100, help='targetnets_updatefreq')


    parser.add_argument('--tensorboard_freq', type=int, default=1, help='tensorboard_freq')
    parser.add_argument('--save_model_freq', type=int, default=100, help='save_model_freq')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint', help='checkpoint_dir')
    
    parser.add_argument('--save_model_dir', type=str, default=exp_name, help='save_model_dir')
    parser.add_argument('--log_dir', type=str, default='logs/pac/'+exp_name, help='logdir')
    parser.add_argument('--load_replay_buffer', type=bool, default=True, help='load_replay_buffer')
    parser.add_argument('--replay_buffer_root', type=str, default='buffer_'+exp_name +'.pt', help='replay_buffer_root')
    parser.add_argument('--is_test', type=int, default=0, help='is_test')
    parser.add_argument('--excel_dir', type=str,default=None, help='excel_dir')
    parser.add_argument('--pac_continue', type=bool, default=True, help='pac_continue')

    # 解析参数
    args = parser.parse_args()
    return args
