from __future__ import annotations
from modules.agents.rnn_agent import RNNAgent
from components.action_selectors import EpsilonGreedyActionSelector
import torch as th
from utils.logging import Logger


# This multi-agent controller shares parameters between agents
class BasicMAC:
    """
    エージェント間でパラメータを共有するマルチエージェントコントローラー

    全エージェントで1つのニューラルネットワークを共有する

    Agent Network (`RNNAgent`) の入出力を制御
    """

    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args

        # 部分観測の次元数
        input_shape = self._get_input_shape(scheme)

        # エージェントネットワークを生成
        self._build_agents(input_shape)

        self.agent_output_type = args.agent_output_type

        # 行動選択アルゴリズムはepsilon-greedy
        self.action_selector = EpsilonGreedyActionSelector(args)

        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False, logger:Logger = None, print_log=False):
        """
        エピソードの開始から現在までの情報を受け取り、行動を選択する
        """
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(
            agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode, logger=logger, print_log=print_log)
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        """

        """
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        agent_outs, self.hidden_states = self.agent(
            agent_inputs, self.hidden_states)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(
                    ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
            if not test_mode:
                # Epsilon floor
                epsilon_action_num = agent_outs.size(-1)
                if getattr(self.args, "mask_before_softmax", True):
                    # With probability epsilon, we will pick an available action uniformly
                    epsilon_action_num = reshaped_avail_actions.sum(
                        dim=1, keepdim=True).float()

                agent_outs = ((1 - self.action_selector.epsilon) * agent_outs
                              + th.ones_like(agent_outs) * self.action_selector.epsilon / epsilon_action_num)

                if getattr(self.args, "mask_before_softmax", True):
                    # Zero out the unavailable actions
                    agent_outs[reshaped_avail_actions == 0] = 0.0

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def init_hidden(self, batch_size):
        """
        RNNの隠れ状態を初期化
        """
        self.hidden_states = self.agent.init_hidden().unsqueeze(
            0).expand(batch_size, self.n_agents, -1)  # bav

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac: BasicMAC):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(
            th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        """
        エージェントネットワークを生成する
        """
        self.agent = RNNAgent(input_shape, self.args)
        pass

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
                inputs.append(batch["actions_onehot"][:, t - 1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(
                0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs * self.n_agents, -1)
                         for x in inputs], dim=1)
        return inputs

    def _get_input_shape(self, scheme):
        """
        エージェントの部分観測の次元数を取得
        """
        # 環境から取得できる"obs_shape"
        input_shape = scheme["obs"]["vshape"]

        if self.args.obs_last_action:
            # Include the agent's last action (one_hot) in the observation
            # 観測にエージェントの前回の行動を含めるか？
            input_shape += scheme["actions_onehot"]["vshape"][0]  # 行動の次元数を追加
        if self.args.obs_agent_id:
            # Include the agent's one_hot id in the observation
            # 観測に全エージェントのOne Hot IDを含めるか？
            # 全エージェントでネットワークを共有するため、どのエージェントの観測かを区別するために必要
            input_shape += self.n_agents

        return input_shape
