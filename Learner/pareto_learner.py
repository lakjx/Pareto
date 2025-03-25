import copy
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from itertools import product
from episode_replaybuffer import EpisodeBatch
from Learner.PPO_learner import CentralVCriticNS
from utils import RunningMeanStd
from einops import rearrange, reduce, repeat
from tensorboardX import SummaryWriter

class OpponentActionPredictor(nn.Module):
    def __init__(self, input_shape, hidden_dim, n_actions, n_agents):
        super(OpponentActionPredictor, self).__init__()
        self.fc1 = nn.Linear(input_shape, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, n_actions * (n_agents - 1))

        self.n_agents = n_agents
        self.n_actions = n_actions
    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        logits = logits.view(*logits.shape[:-1], self.n_agents-1, self.n_actions)

        return F.softmax(logits, dim=-1)
        
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
class PACCriticNS(nn.Module):
    def __init__(self, scheme, args):
        super(PACCriticNS, self).__init__()
        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents

        input_shape = self._get_input_shape(scheme)
        self.output_type = "q"

        # Set up network layers
        self.critics = [MLP(input_shape, args.hidden_dim, self.n_actions) for _ in range(self.n_agents)]

        self.device = "cuda" if args.use_cuda else "cpu"

        if args.pac_continue :
            self.oppo_pred = OpponentActionPredictor(
                input_shape=args.state_dim,
                hidden_dim=args.hidden_dim,
                n_actions=args.n_actions,
                n_agents=args.n_agents
            )
            self.oppo_pred.to(self.device)
    def forward(self, batch, t=None, compute_all=False, pac_continue=False):
        if not pac_continue:
            if compute_all:
                inputs, bs, max_t, other_actions = self._build_inputs_all(batch, t=t)
            else:
                inputs, bs, max_t, other_actions = self._build_inputs_cur(batch, t=t)
        else:
            inputs = []
            inputs.append(batch["state"].unsqueeze(2).repeat(1, 1, self.n_agents, 1))
            other_actions = self._pred_opponent_actions(batch, t=t)
            
            inputs.append(other_actions)
            inputs = th.cat(inputs, dim=-1)
        
        qs = []
        for i in range(self.n_agents):
            q = self.critics[i](inputs[:, :, i]).unsqueeze(2)
            qs.append(q)
        return th.cat(qs, dim=2), other_actions

    def _pred_opponent_actions(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)

        inputs = []
        inputs.append(batch["state"][:, ts].unsqueeze(2).repeat(1, 1, self.n_agents, 1))
        inputs = th.cat(inputs, dim=-1)
        other_action_probs = self.oppo_pred(inputs)
        other_action_indices = th.distributions.Categorical(probs=other_action_probs).sample()

        #One-Hot transform
        other_actions = th.zeros_like(other_action_probs)
        other_actions.scatter_(-1, other_action_indices.unsqueeze(-1), 1)
        
        return other_actions.view(bs, max_t, self.n_agents, -1)

    def _gen_all_other_actions(self, batch, bs, max_t):

        other_agents_actions = generate_other_actions(self.n_actions, self.n_agents, self.device)
        n_other_actions = other_agents_actions.shape[0]
        other_agents_actions = repeat(other_agents_actions, "e f -> n s a e f", n=bs, s=max_t, a=self.n_agents)
        return other_agents_actions

    def _gen_subsample_other_actions(self, batch, bs, max_t, sample_size):

        avail_actions = batch["avail_actions"]

        # ALL AVAIL ACTIONS ARE ZERO IF EPISODE HAS TERMINATED
        probs =avail_actions/avail_actions.sum(dim=-1).unsqueeze(-1)
        probs = th.nan_to_num(probs, nan=1.0/avail_actions.size(-1))

        avail_dist = th.distributions.OneHotCategorical(probs=probs)
        sample = avail_dist.sample([sample_size])
        samples = []
        for i in range(self.n_agents):
            samples.append(th.cat([sample[:, :, :, j, :] for j in range(self.n_agents) if j != i], dim=-1))
        samples = th.stack(samples)
        samples = rearrange(samples, "i j k l m -> k l i j m")
        return samples

    def _build_inputs_all(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1

        ts = slice(None) if t is None else slice(t, t+1)
        inputs = []
        # state
        inputs.append(batch["state"][:, ts].unsqueeze(2).repeat(1, 1, self.n_agents, 1))

        # observations
        if self.args.obs_individual_obs:
            inputs.append(batch["obs"][:, ts].view(bs, max_t, -1).unsqueeze(2).repeat(1, 1, self.n_agents, 1))

        # last actions
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, 0:1]).view(bs, max_t, 1, -1))
            elif isinstance(t, int):
                inputs.append(batch["actions_onehot"][:, slice(t-1, t)].view(bs, max_t, 1, -1))
            else:
                last_actions = th.cat([th.zeros_like(batch["actions_onehot"][:, 0:1]), batch["actions_onehot"][:, :-1]], dim=1)
                last_actions = last_actions.view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1)
                inputs.append(last_actions)

        inputs = th.cat(inputs, dim=-1)

        other_actions = self._gen_all_other_actions(batch, bs, max_t)

        n_other_actions = other_actions.size(3)

        inputs = repeat(inputs, "n s a f -> n s a e f", e=n_other_actions)
        inputs = th.cat((inputs, other_actions), dim=-1)

        # print(inputs.shape)

        return inputs, bs, max_t, other_actions

    def _build_inputs_cur(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1

        ts = slice(None) if t is None else slice(t, t+1)
        inputs = []
        # state
        inputs.append(batch["state"][:, ts].unsqueeze(2).repeat(1, 1, self.n_agents, 1))

        # observations
        if self.args.obs_individual_obs:
            inputs.append(batch["obs"][:, ts].view(bs, max_t, -1).unsqueeze(2).repeat(1, 1, self.n_agents, 1))

        # last actions
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, 0:1]).view(bs, max_t, 1, -1))
            elif isinstance(t, int):
                inputs.append(batch["actions_onehot"][:, slice(t-1, t)].view(bs, max_t, 1, -1))
            else:
                last_actions = th.cat([th.zeros_like(batch["actions_onehot"][:, 0:1]), batch["actions_onehot"][:, :-1]], dim=1)
                last_actions = last_actions.view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1)
                inputs.append(last_actions)

        actions = []
        for i in range(self.n_agents):
            actions.append(th.cat([batch["actions_onehot"][:, :, j].unsqueeze(2) for j in range(self.n_agents)
                                   if j != i], dim=-1))
        actions = th.cat(actions, dim=2)
        inputs.append(actions)
        # inputs.append()
        inputs = th.cat(inputs, dim=-1)
        return inputs, bs, max_t, actions

    def _get_input_shape(self, scheme):
        # state
        input_shape = scheme["state"]["vshape"]
        # observations
        if self.args.obs_individual_obs:
            input_shape += scheme["obs"]["vshape"] * self.n_agents
        # last actions
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0] * self.n_agents
        input_shape += self.n_actions * (self.n_agents - 1)
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

class Pareto_Learner:
    def __init__(self, mac, args, scheme):
        log_dir = args.log_dir
        self.writer = SummaryWriter(log_dir=log_dir)
        self.args = args
        self.scheme = scheme
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions

        self.mac = mac
        self.agent_params = list(mac.parameters())
        self.agent_optimiser = Adam(params=self.agent_params, lr=args.lr, weight_decay=args.optim_weight_decay)

        self.critic = PACCriticNS(scheme, args)
        self.target_critic = copy.deepcopy(self.critic)
        self.state_value = CentralVCriticNS(scheme, args)

        self.critic_params = list(self.critic.parameters())+list(self.state_value.parameters())
        self.critic_optimiser = Adam(params=self.critic_params, lr=args.lr, weight_decay=args.optim_weight_decay)

        self.last_target_update_step = 0
        self.critic_training_steps = 0

        device = "cuda" if args.use_cuda else "cpu"
        self.ret_ms = RunningMeanStd(shape=(self.n_agents, ), device=device)
    
    def train(self, batch: EpisodeBatch):
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        mask = mask.repeat(1, 1, self.n_agents)

        critic_mask = mask.clone()

        mac_outs=[]
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length-1):
            agent_outs = self.mac.forward(batch, t=t)
            mac_outs.append(agent_outs)
        mac_outs = th.stack(mac_outs, dim=1)

        pi =mac_outs
        advantages,critic_train_stats = self.train_critic(self.critic, self.target_critic, batch, rewards,critic_mask)
        actions = actions[:, :-1]
        advantages = advantages.detach()
        #Calculate the policy gradient loss

        if self.args.action_is_mix:
                # 分解 mac_out 和 actions
                pi_discrete, pi_continuous_mean, pi_continuous_std = mac_outs.split([5, 3, 3], dim=-1)
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
                pi = mac_outs[..., :5]
        else:
            pi[mask == 0] = 1.0
            pi_taken = th.gather(pi, dim=3, index=actions).squeeze(3)
            log_pi_taken = th.log(pi_taken + 1e-8)

        entropy = -th.sum(pi * th.log(pi + 1e-8), dim=-1)

        pg_loss = -((advantages * log_pi_taken + self.args.entropy_coef * entropy) * mask).sum() / mask.sum()

        #Optimize the policy
        self.agent_optimiser.zero_grad()
        pg_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
        self.agent_optimiser.step()

        self.critic_training_steps += 1
        target_update_interval_or_tau = 0.01
        self._update_targets_soft(target_update_interval_or_tau)

        if self.critic_training_steps % self.args.tensorboard_freq == 0:
            ts_logged = len(critic_train_stats["critic_loss"])
            ts_logged = len(critic_train_stats["critic_loss"])
            self.writer.add_scalar("critic_loss", sum(critic_train_stats["critic_loss"]) / ts_logged, self.critic_training_steps)
            self.writer.add_scalar("critic_grad_norm", sum(critic_train_stats["critic_grad_norm"]) / ts_logged, self.critic_training_steps)
            self.writer.add_scalar("td_error_abs", sum(critic_train_stats["td_error_abs"]) / ts_logged, self.critic_training_steps)
            # self.writer.add_scalar("target_mean", sum(critic_train_stats["target_mean"]) / ts_logged, self.critic_training_steps)
            self.writer.add_scalar("q_taken_mean", sum(critic_train_stats["q_taken_mean"]) / ts_logged, self.critic_training_steps)
            for id_ in range(self.n_agents):
                self.writer.add_scalar("target_reward{}".format(id_+1), sum(critic_train_stats["target_reward{}".format(id_+1)]) / ts_logged, self.critic_training_steps)
                self.writer.add_scalar("advantage_mean{}".format(id_+1), (advantages[:, :, id_]).sum().item(), self.critic_training_steps)
            
            self.writer.add_scalar("pg_loss", pg_loss.item(), self.critic_training_steps)
            self.writer.add_scalar("agent_grad_norm", grad_norm, self.critic_training_steps)
            # self.writer.add_scalar("advantage_mean",(advantages * mask).sum().item() / mask.sum().item(), self.critic_training_steps)
    
    def expectile_loss(self, diff, expectile):
        weight = th.where(diff > 0, th.tensor(expectile), th.tensor(1 - expectile))
        return weight * (diff ** 2)

    def train_critic(self, critic, target_critic, batch, rewards, mask):
        actions = batch["actions"]
        with th.no_grad():
            if hasattr(target_critic,'oppo_pred'):
                target_qvals, other_actions = target_critic(batch, pac_continue=True)
            else:
                target_qvals = target_critic(batch, compute_all=True)[0][:, :-1]
                target_qvals = target_qvals.max(dim=3)[0]
           
        target_qvals = th.gather(target_qvals, -1, actions[:, :-1]).squeeze(-1)

        target_returns = self.nstep_returns(rewards, mask, target_qvals, self.args.q_nstep)
        if self.args.standardise_rewards:
            self.ret_ms.update(target_returns)
            target_returns = (target_returns - self.ret_ms.mean) / th.sqrt(self.ret_ms.var)

        running_log = {
            "critic_loss": [],
            "critic_grad_norm": [],
            "td_error_abs": [],
            "q_taken_mean": [],
        }
        for id_ in range(self.n_agents):
            running_log["target_reward{}".format(id_+1)] = []

        actions = batch["actions"][:, :-1]
        q = critic(batch,pac_continue = True)[0][:, :-1] # if hasattr(target_critic,'oppo_pred') else critic(batch)[0][:, :-1]
        v = self.state_value(batch)[:, :-1].squeeze(-1)

        q_current = th.gather(q, -1, actions).squeeze(-1)
        td_error = target_returns.detach() - q_current
        mask_td_error = td_error * mask
        # loss = 0.5 * (mask_td_error ** 2).sum() / mask.sum()
        loss = self.expectile_loss(mask_td_error, self.args.expectile).sum() / mask.sum()

        td_error_v = target_returns.detach() - v
        mask_td_error_v = td_error_v * mask
        # loss += 0.5 * (mask_td_error_v ** 2).sum() / mask.sum()
        loss += self.expectile_loss(mask_td_error_v, self.args.expectile).sum() / mask.sum()

        # compute the maximum Q-value and the joint action of the other agents that results in this Q-value
        if hasattr(critic,'oppo_pred'):
            q_all = critic(batch, pac_continue=True)[0][:, :-1]
        else:
            q_all = critic(batch, compute_all=True)[0][:, :-1]
            q_all = q_all.max(dim=3)[0]
        
        q_all = th.gather(q_all, -1, actions).squeeze(-1)

        advantage = q_all.detach() - v.detach()

        #加入动作预测的损失
        if hasattr(critic,'oppo_pred'):
            # other_actions = other_actions[:, :-1]
            other_actions_probs = critic.oppo_pred(batch["state"][:, :-1].unsqueeze(2).repeat(1, 1, self.n_agents, 1))
            
            actual_action_indices = []
            for id_ in range(self.n_agents):
                other_agents = [j for j in range(self.n_agents) if j != id_]
                agent_indices = []
                for j, other_id in enumerate(other_agents):
                    # 获取其他智能体的实际动作索引
                    indices = batch["actions"][:, :-1, other_id]
                    agent_indices.append(indices)
                actual_action_indices.append(th.stack(agent_indices, dim=-1))
            actual_action_indices = th.stack(actual_action_indices, dim=2).squeeze(3)  # [bs, max_t-1, n_agents, n_agents-1]

            # actual_actions = batch["actions_onehot"][:, :-1].view(batch.batch_size, batch.max_seq_length-1,self.n_agents, -1)
            # # 通过actual_actions转化得到真实的actual_other_actions
            # actual_other_actions = []
            # for id_ in range(self.n_agents):
            #     single_actual_other_actions = th.cat([actual_actions[:, :, j, :] for j in range(self.n_agents) if j != id_], dim=-1) 
            #     actual_other_actions.append(single_actual_other_actions)
            # actual_other_actions = th.stack(actual_other_actions, dim=2)
            
            pre_action_loss = 0
            for id_ in range(self.n_agents):
                for j in range(self.n_agents-1):
                    pred = other_actions_probs[:, :, id_, j]
                    target = actual_action_indices[:, :, id_, j]
                    pre_action_loss += F.cross_entropy(pred.reshape(-1, self.n_actions), target.reshape(-1))
            pre_action_loss /= (self.n_agents * (self.n_agents-1))
            loss += 0.45 * pre_action_loss

        self.critic_optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
        self.critic_optimiser.step()

        running_log["critic_loss"].append(loss.item())
        running_log["critic_grad_norm"].append(grad_norm.item())
        mask_elems = mask.sum().item()
        running_log["td_error_abs"].append((mask_td_error.abs().sum().item() / mask_elems))
        running_log["q_taken_mean"].append((q_current * mask).sum().item() / mask_elems)
        
        for id_ in range(self.n_agents):
            running_log["target_reward{}".format(id_+1)].append((target_returns[:, :, id_]).sum().item())

        return advantage, running_log
    
    def nstep_returns(self, rewards, mask, values, nsteps):
        rewards = rewards.squeeze(3)
        nstep_values = th.zeros_like(values)
        for t_start in range(rewards.size(1)):
            nstep_return_t = th.zeros_like(values[:, 0])
            for step in range(nsteps + 1):
                t = t_start + step
                if t >= rewards.size(1):
                    break
                elif step == nsteps:
                    nstep_return_t += self.args.gamma ** (step) * values[:, t] * mask[:, t]
                else:
                    nstep_return_t += self.args.gamma ** (step) * rewards[:, t] * mask[:, t]
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
        self.mac.cuda()
        self.critic.cuda()
        self.target_critic.cuda()
        self.state_value.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.critic.state_dict(), "{}/critic.th".format(path))
        th.save(self.agent_optimiser.state_dict(), "{}/agent_opt.th".format(path))
        th.save(self.critic_optimiser.state_dict(), "{}/critic_opt.th".format(path))
    def load_models(self, path):
        device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        self.mac.load_models(path)
        self.critic.load_state_dict(th.load("{}/critic.th".format(path), map_location=device))
        # Not quite right but I don't want to save target networks
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.agent_optimiser.load_state_dict(
            th.load("{}/agent_opt.th".format(path), map_location=device))
        self.critic_optimiser.load_state_dict(
            th.load("{}/critic_opt.th".format(path), map_location=device))

        # Move optimizer's state tensors to the same device as model parameters
        for state in self.critic_optimiser.state.values():
            for k, v in state.items():
                if isinstance(v, th.Tensor):
                    state[k] = v.to(device)
        for state in self.agent_optimiser.state.values():
            for k, v in state.items():
                if isinstance(v, th.Tensor):
                    state[k] = v.to(device)
    # def load_models(self, path):
    #     self.mac.load_models(path)
    #     self.critic.load_state_dict(th.load("{}/critic.th".format(path), map_location=lambda storage, loc: storage))
    #     # Not quite right but I don't want to save target networks
    #     self.target_critic.load_state_dict(self.critic.state_dict())
    #     self.agent_optimiser.load_state_dict(
    #         th.load("{}/agent_opt.th".format(path), map_location=lambda storage, loc: storage))
    #     self.critic_optimiser.load_state_dict(
    #         th.load("{}/critic_opt.th".format(path), map_location=lambda storage, loc: storage))
   
def generate_other_actions(n_actions, n_agents, device):
    other_acts = [
        th.cat(x) for x in product(*[th.eye(n_actions, device=device) for _ in range(n_agents - 1)])
    ]
    other_acts = th.stack(other_acts)
    return other_acts