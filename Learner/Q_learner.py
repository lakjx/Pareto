import copy
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from episode_replaybuffer import EpisodeBatch
from utils import RunningMeanStd
from tensorboardX import SummaryWriter

class QMixer(nn.Module):
    def __init__(self, args):
        super(QMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_dim))

        self.embed_dim = args.mixing_embed_dim

        if getattr(args, "hypernet_layers", 1) == 1:
            self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
            self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
        elif getattr(args, "hypernet_layers", 1) == 2:
            hypernet_embed = self.args.hypernet_embed
            self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim * self.n_agents))
            self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim))
        elif getattr(args, "hypernet_layers", 1) > 2:
            raise Exception("Sorry >2 hypernet layers is not implemented!")
        else:
            raise Exception("Error setting number of hypernet layers.")

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)
        # First layer
        w1 = th.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(th.bmm(agent_qs, w1) + b1)
        # Second layer
        w_final = th.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)
        # Compute final output
        y = th.bmm(hidden, w_final) + v
        # Reshape and return
        q_tot = y.view(bs, -1, 1)
        return q_tot
class VDNMixer(nn.Module):
    def __init__(self):
        super(VDNMixer, self).__init__()

    def forward(self, agent_qs, batch):
        return th.sum(agent_qs, dim=2, keepdim=True)
       
class Q_Learner:
    def __init__(self, mac,args):
        self.mac = mac
        self.args = args
        self.writer = SummaryWriter(log_dir=args.log_dir)
        self.target_mac = copy.deepcopy(mac)

        self.params = list(mac.parameters())
        self.mixer = None

        if args.mixer is not None:
            if args.mixer == "qmix":
                self.mixer = QMixer(args)
            elif args.mixer == "vdn":
                self.mixer = VDNMixer()
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)
        # Optimizer
        self.optimiser = th.optim.Adam(params=self.params, lr=args.lr, weight_decay=args.optim_weight_decay)

        self.training_steps = 0
        self.last_target_update_step = 0
        
        device = th.device("cuda" if th.cuda.is_available() else "cpu")
        if self.args.standardise_rewards:
            self.ret_rms = RunningMeanStd(shape=(1,),device=device)
    
    def train(self, batch: EpisodeBatch):
        # Get the relevant quantities  batchsize*max_seq_length*n_agents*n_actions
        rewards = batch["reward"][:,:-1]
        actions = batch["actions"][:,:-1]
        terminated = batch["terminated"][:,:-1].float()

        #奖励归一化
        if self.args.standardise_rewards:
            rewards = (rewards - self.ret_rms.mean) / th.sqrt(self.ret_rms.var)

        #计算Q估计值
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t_ep=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)
        #选择动作的Q值
        chosen_action_qvals = th.gather(mac_out[:,:-1], dim=3, index=actions).squeeze(3)

        #计算目标Q值
        taget_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.target_mac.forward(batch, t_ep=t)
            taget_mac_out.append(agent_outs)
        #不需要第一个Q值
        taget_mac_out = th.stack(taget_mac_out[1:], dim=1)

        # Max over target Q-Values
        if self.args.doubleQ:
            mac_out_detach = mac_out.clone().detach()
            current_max_actions = mac_out_detach[:,1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(taget_mac_out, 3, current_max_actions).squeeze(3)
        else:
            target_max_qvals = taget_mac_out.max(dim=3)[0]

        # mixer
        if self.args.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:,:-1])
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:,:-1])

        #Cualculate 1-step Q-Learning targets
        if self.args.nstep_return > 1:
            nstep_returns = self.nstep_returns(rewards, target_max_qvals, self.args.nstep_return)
            targets = nstep_returns
        else:
            targets = th.mean(rewards,dim = 2) + self.args.gamma * (1 - terminated) * target_max_qvals.detach()

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())
        mask = (1 - terminated)
        masked_td_error = td_error * mask
        
        loss = (masked_td_error ** 2).sum() / mask.sum()

        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        self.training_steps += 1
        if self.training_steps % self.args.targetnets_updatefreq == 0:
            self._update_targets_hard()
        
        # Tensorboard logging
        if self.training_steps % self.args.tensorboard_freq == 0:
            self.writer.add_scalar("Loss", loss.item(), self.training_steps)
            self.writer.add_scalar("Grad Norm", grad_norm.item(), self.training_steps)
            self.writer.add_scalar("TD Error", td_error.mean().item(), self.training_steps)
            self.writer.add_scalar("Q-Value", chosen_action_qvals.mean().item()/self.args.n_agents, self.training_steps)
            # Add more logging here if needed

    def _update_targets_hard(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())

    def _update_targets_soft(self, tau):
        for target_param, param in zip(self.target_mac.parameters(), self.mac.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        if self.mixer is not None:
            for target_param, param in zip(self.target_mixer.parameters(), self.mixer.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()
    def nstep_returns(self, rewards, values, nsteps):
        nstep_values = th.zeros_like(values[:, :-1])
        for t_start in range(rewards.size(1)):
            nstep_return_t = th.zeros_like(values[:, 0])
            for step in range(nsteps + 1):
                t = t_start + step
                if t >= rewards.size(1):
                    break
                elif step == nsteps:
                    nstep_return_t += self.args.gamma ** (step) * values[:, t]
                elif t == rewards.size(1) - 1 and self.args.add_value_last_step:
                    nstep_return_t += self.args.gamma ** (step) * rewards[:, t]
                    nstep_return_t += self.args.gamma ** (step + 1) * values[:, t + 1]
                else:
                    nstep_return_t += self.args.gamma ** (step) * rewards[:, t]
            nstep_values[:, t_start, :] = nstep_return_t
        return nstep_values