import copy
import torch as th
import torch.nn.functional as F
from torch.optim import Adam
from torch import nn
from episode_replaybuffer import EpisodeBatch
from utils import RunningMeanStd
from tensorboardX import SummaryWriter
class MLP(nn.Module):
    def __init__(self, input_shape, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_shape, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q

class CentralVCriticNS(nn.Module):
    def __init__(self, scheme, args):
        super(CentralVCriticNS, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents

        input_shape = self._get_input_shape(scheme)
        self.output_type = "v"

        # Set up network layers
        self.critics = [MLP(input_shape, args.hidden_dim, 1) for _ in range(self.n_agents)]

    def forward(self, batch, t=None):
        inputs, bs, max_t = self._build_inputs(batch, t=t)
        qs = []
        for i in range(self.n_agents):
            q = self.critics[i](inputs)
            qs.append(q.view(bs, max_t, 1, -1))
        q = th.cat(qs, dim=2)
        return q

    def _build_inputs(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)
        inputs = []
        # state
        inputs.append(batch["state"][:, ts])

        # observations
        if self.args.obs_individual_obs:
            inputs.append(batch["obs"][:, ts].view(bs, max_t, -1))

        if self.args.obs_last_action:
            # last actions
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, 0:1]).view(bs, max_t, 1, -1))
            elif isinstance(t, int):
                inputs.append(batch["actions_onehot"][:, slice(t-1, t)].view(bs, max_t, 1, -1))
            else:
                last_actions = th.cat([th.zeros_like(batch["actions_onehot"][:, 0:1]), batch["actions_onehot"][:, :-1]], dim=1)
                last_actions = last_actions.view(bs, max_t, 1, -1)
                inputs.append(last_actions)

        inputs = th.cat([x.reshape(bs * max_t, -1) for x in inputs], dim=1)
        return inputs, bs, max_t

    def _get_input_shape(self, scheme):
        # state
        input_shape = scheme["state"]["vshape"]
        # observations
        if self.args.obs_individual_obs:
            input_shape += scheme["obs"]["vshape"]
        # last actions
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0] * self.n_agents

        return input_shape

    def parameters(self):
        params = list(self.critics[0].parameters())
        for i in range(1, self.n_agents):
            params += list(self.critics[i].parameters())
        return params

    def state_dict(self):
        return [a.state_dict() for a in self.critics]

    def load_state_dict(self, state_dict):
        for i, a in enumerate(self.critics):
            a.load_state_dict(state_dict[i])

    def cuda(self):
        for c in self.critics:
            c.cuda()

class PPO_Learner:
    def __init__(self, mac,scheme,args):
        log_dir = args.log_dir+"/PPO"
        self.writer = SummaryWriter(log_dir=log_dir)
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        
        self.mac = mac
        self.old_mac = copy.deepcopy(mac)
        self.agent_params = list(mac.parameters())
        self.agent_optimiser = Adam(params=self.agent_params, lr=args.lr)

        self.critic = CentralVCriticNS(scheme,args)
        self.target_critic = copy.deepcopy(self.critic)

        self.critic_params = list(self.critic.parameters())
        self.critic_optimiser = Adam(params=self.critic_params, lr=args.lr)

        self.last_target_update_step = 0
        self.critic_training_steps = 0

        device = "cuda" if args.use_cuda else "cpu"

        self.ret_ms = RunningMeanStd(shape=(1,), device=device)

    def train(self, batch: EpisodeBatch):
        # Get the relevant quantities  batchsize*max_seq_length*n_agents*n_actions
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()

        if self.args.standardise_rewards:
            rewards = (rewards - self.ret_ms.mean) / th.sqrt(self.ret_ms.var)

        # Get the critic outputs
        old_mac_out = []
        self.old_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length - 1):
            agent_outs = self.old_mac.forward(batch, t=t)
            old_mac_out.append(agent_outs)
        old_mac_out = th.stack(old_mac_out, dim=1)  # Concat over time
        old_pi = old_mac_out

        if self.args.action_is_mix:
            # 分解 old_mac_out 和 actions
            old_pi_discrete, old_pi_continuous_mean, old_pi_continuous_std = old_mac_out.split([5, 3, 3], dim=-1)
            discrete_actions, continuous_actions = actions.split([1, 3], dim=-1)
            #处理离散动作
            old_pi_discrete_taken = th.gather(old_pi_discrete, dim=3, index=discrete_actions.long()).squeeze(3)
            # 处理连续动作
            old_pi_continuous_std = th.clamp(old_pi_continuous_std, min=1e-6, max=1)
            old_pi_continuous_distribution = th.distributions.Normal(old_pi_continuous_mean, old_pi_continuous_std)
            old_pi_continuous_taken = th.exp(old_pi_continuous_distribution.log_prob(continuous_actions))
            # 计算混合动作的概率
            old_pi_taken = old_pi_discrete_taken * old_pi_continuous_taken.prod(dim=-1)
            old_log_pi_taken = th.log(old_pi_taken + 1e-10)
        else:
            old_pi_taken = th.gather(old_pi, dim=3, index=actions.long()).squeeze(3)
            old_log_pi_taken = th.log(old_pi_taken + 1e-10)

        # Train the critic
        for _ in range(self.args.ppo_epochs):
            mac_out = []
            self.mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length - 1):
                agent_outs = self.mac.forward(batch, t=t)
                mac_out.append(agent_outs)
            mac_out = th.stack(mac_out, dim=1)
            
            pi = mac_out
            if self.args.action_is_mix:
                # 分解 mac_out 和 actions
                pi_discrete, pi_continuous_mean, pi_continuous_std = mac_out.split([5, 3, 3], dim=-1)
                discrete_actions, continuous_actions = actions.split([1, 3], dim=-1)
                #处理离散动作
                pi_discrete_taken = th.gather(pi_discrete, dim=3, index=discrete_actions.long()).squeeze(3)
                # 处理连续动作
                pi_continuous_std = th.clamp(pi_continuous_std, min=1e-6, max=1)
                pi_continuous_distribution = th.distributions.Normal(pi_continuous_mean, pi_continuous_std)
                pi_continuous_taken = th.exp(pi_continuous_distribution.log_prob(continuous_actions))
                # 计算混合动作的概率
                pi_taken = pi_discrete_taken * pi_continuous_taken.prod(dim=-1)
                log_pi_taken = th.log(pi_taken + 1e-10)
                pi = mac_out[..., :5]
            else:
                pi_taken = th.gather(pi, dim=3, index=actions.long()).squeeze(3)
                log_pi_taken = th.log(pi_taken + 1e-10)

            advantages, critic_train_stats = self.train_critic_sequential(self.critic, self.target_critic, batch, rewards)
            advantages = advantages.detach()

            ratio = th.exp(log_pi_taken - old_log_pi_taken.detach())
            surr1 = ratio * advantages
            surr2 = th.clamp(ratio, 1 - self.args.ppo_clip_param, 1 + self.args.ppo_clip_param) * advantages

            entropy = -th.sum(pi * th.log(pi + 1e-10), dim=-1)+pi_continuous_distribution.entropy().sum(dim=-1)
            pg_loss = (-th.min(surr1, surr2)+ self.args.entropy_coef * entropy).mean()

            # Optimize agents
            self.agent_optimiser.zero_grad()
            pg_loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
            self.agent_optimiser.step()
        
        self.old_mac.load_state(self.mac)

        self.critic_training_steps += 1
        if self.args.target_update_interval_or_tau > 1 and (
                self.critic_training_steps - self.last_target_update_step) / self.args.target_update_interval_or_tau >= 1.0:
            self._update_targets_hard()
            self.last_target_update_step = self.critic_training_steps
        
        if self.critic_training_steps % self.args.tensorboard_freq == 0:
            ts_logged = len(critic_train_stats["critic_loss"])
            self.writer.add_scalar("critic_loss", sum(critic_train_stats["critic_loss"]) / ts_logged, self.critic_training_steps)
            self.writer.add_scalar("critic_grad_norm", sum(critic_train_stats["critic_grad_norm"]) / ts_logged, self.critic_training_steps)
            self.writer.add_scalar("td_error_abs", sum(critic_train_stats["td_error_abs"]) / ts_logged, self.critic_training_steps)
            self.writer.add_scalar("target_mean", sum(critic_train_stats["target_mean"]) / ts_logged, self.critic_training_steps)
            self.writer.add_scalar("q_taken_mean", sum(critic_train_stats["q_taken_mean"]) / ts_logged, self.critic_training_steps)

            self.writer.add_scalar("pg_loss", pg_loss.item(), self.critic_training_steps)
            self.writer.add_scalar("agent_grad_norm", grad_norm.item(), self.critic_training_steps)
            self.writer.add_scalar("advantages", advantages.mean().item(), self.critic_training_steps)
            self.writer.add_scalar("pi_max",pi.max(dim=-1)[0].mean().item(),self.critic_training_steps)

    def train_critic_sequential(self, critic, target_critic, batch, rewards):
        # Optimise critic
        with th.no_grad():
            target_vals = target_critic(batch)
            target_vals = target_vals.squeeze(3)

        target_returns = self.nstep_returns(rewards, target_vals, self.args.q_nstep)
       
        # if self.args.standardise_returns:
        #     self.ret_ms.update(target_returns)
        #     target_returns = (target_returns - self.ret_ms.mean) / th.sqrt(self.ret_ms.var)

        running_log = {
            "critic_loss": [],
            "critic_grad_norm": [],
            "td_error_abs": [],
            "target_mean": [],
            "q_taken_mean": [],
        }

        v = critic(batch)[:, :-1].squeeze(3)
        td_error = (target_returns.detach() - v)
        # masked_td_error = td_error * mask
        loss = (td_error ** 2).mean()

        self.critic_optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
        self.critic_optimiser.step()

        running_log["critic_loss"].append(loss.item())
        running_log["critic_grad_norm"].append(grad_norm.item())
        # mask_elems = mask.sum().item()
        running_log["td_error_abs"].append((td_error.abs().mean()).item())
        running_log["q_taken_mean"].append(v.mean().item())
        running_log["target_mean"].append((target_returns.mean()).item())

        return td_error, running_log

    def nstep_returns(self, rewards, values, nsteps):
        # rewards =th.mean(rewards,dim = 2)
        rewards = rewards.squeeze(3)
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

    def _update_targets(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def _update_targets_hard(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def _update_targets_soft(self, tau):
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def cuda(self):
        self.old_mac.cuda()
        self.mac.cuda()
        self.critic.cuda()
        self.target_critic.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.critic.state_dict(), "{}/critic.th".format(path))
        th.save(self.agent_optimiser.state_dict(), "{}/agent_opt.th".format(path))
        th.save(self.critic_optimiser.state_dict(), "{}/critic_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.critic.load_state_dict(th.load("{}/critic.th".format(path), map_location=lambda storage, loc: storage))
        # Not quite right but I don't want to save target networks
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.agent_optimiser.load_state_dict(
            th.load("{}/agent_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.critic_optimiser.load_state_dict(
            th.load("{}/critic_opt.th".format(path), map_location=lambda storage, loc: storage)) 