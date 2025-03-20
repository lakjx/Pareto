import torch as th
from Learner.RnnAgent import RnnAgent, EpsilonGreedyActionSelector, NoSharedRnnAgent,SoftPoliciesSelector

class BasicMac:
    def __init__(self, args,scheme):
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type
        self.action_selector = EpsilonGreedyActionSelector(args)

        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep,bs=slice(None), test_mode=False):
        avail_actions = th.ones(ep_batch.batch_size, self.n_agents, self.args.n_actions).to(ep_batch.device)
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], test_mode=test_mode)
        return chosen_actions
    
    def forward(self, ep_batch, t_ep, test_mode=False):
        inputs = self._build_inputs(ep_batch, t_ep)
        
        agent_outs, self.hidden_states = self.agent(inputs, self.hidden_states)

        if self.agent_output_type == "pi_logits":
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
        
        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)
    
    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav
    def _build_agents(self, input_shape):
        self.agent = RnnAgent(input_shape, self.args)
    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
        return inputs
    
    def _get_input_shape(self, scheme):
            input_shape = scheme["obs"]["vshape"]
            if self.args.obs_last_action:
                input_shape += scheme["actions_onehot"]["vshape"][0]
            if self.args.obs_agent_id:
                input_shape += self.n_agents
            return input_shape

    def parameters(self):
        return self.agent.parameters()
    def cuda(self):
        self.agent.cuda()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))
        
    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

class NoSharedMac:
    def __init__(self, args, scheme):
        self.n_agents = args.n_agents
        self.args = args
        self.dis_dim = args.n_clients

        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type

        self.action_selector = SoftPoliciesSelector(args)

        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, bs=slice(None), test_mode=False):
        avail_actions = th.ones(ep_batch.batch_size, self.n_agents, self.args.n_actions).to(ep_batch.device)
        agent_outputs = self.forward(ep_batch, t_ep)
        if self.args.action_is_mix:
            # 分离离散动作和连续动作
            discrete_logits = agent_outputs[..., :self.dis_dim]
            continuous_mean_std = agent_outputs[..., self.dis_dim:]
            # 对离散动作进行采样
            discrete_actions = self.action_selector.select_action(discrete_logits[bs], test_mode=test_mode)
            # 对连续动作进行采样
            continuous_action_mean = continuous_mean_std[..., :continuous_mean_std.size(-1) // 2]
            continuous_action_std = continuous_mean_std[..., continuous_mean_std.size(-1) // 2:]
            continuous_action_std = th.clamp(continuous_action_std, min=1e-6, max=1)
            normal_distribution = th.distributions.Normal(continuous_action_mean, continuous_action_std)
            continuous_actions = normal_distribution.sample()

            # 将离散动作和连续动作合并
            chosen_actions = th.cat([discrete_actions.unsqueeze(-1), continuous_actions], dim=-1)
        else:
            chosen_actions = self.action_selector.select_action(agent_outputs[bs], test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        inputs = self._build_inputs(ep_batch, t)
        device = self.hidden_states.device
        agent_outs, self.hidden_states = self.agent(inputs.to(device), self.hidden_states)
        
        if self.agent_output_type == "pi_logits":
            if self.args.action_is_mix:
                # 获取离散动作的部分
                softmax_outs = th.nn.functional.softmax(agent_outs.narrow(-1, 0, self.dis_dim), dim=-1)
                # 获取剩余的部分
                remaining_outs = agent_outs.narrow(-1, self.dis_dim, agent_outs.size(-1) - self.dis_dim)
                # 合并
                agent_outs = th.cat([softmax_outs, remaining_outs], dim=-1)
            else:
                agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, -1, -1)  # bav

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=th.device("cuda")))

    def _build_agents(self, input_shape):
        self.agent = NoSharedRnnAgent(input_shape, self.args)
    
    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents
        return input_shape
    
    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
        return inputs