import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
import torch as th
from torch.optim import RMSprop, Adam
from controllers.basic_controller import BasicMAC
from utils.logging import Logger
import wandb
import matplotlib.pyplot as plt
import seaborn as sns


class QLearner:
    """
    MAC(Agent Network)とMixingNetworkをまとめて、ネットワークの学習を行う
    """

    def __init__(self, mac: BasicMAC, scheme, logger: Logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                pass
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                pass
                self.mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        # self.optimiser = RMSprop(
        #     params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.optimiser = Adam(params=self.params, lr=args.lr)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        """
        バッチを用いてネットワークを学習 (DDQN)
        """
        # Get the relevant quantities
        # バッチからデータを取り出す

        # rewards shape: torch.Size([バッチサイズ, (最大)エピソード長-1, 1])
        rewards = batch["reward"][:, :-1]

        # actions shape: torch.Size([バッチサイズ, (最大)エピソード長-1, エージェント数, 1])
        actions = batch["actions"][:, :-1]

        # terminated shape: torch.Size([バッチサイズ, (最大)エピソード長-1, 1])
        terminated = batch["terminated"][:, :-1].float()

        # mask shape: torch.Size([バッチサイズ, (最大)エピソード長-1, 1])
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        # avail_actions shape: torch.Size([バッチサイズ, (最大)エピソード長, エージェント数, 行動数])
        avail_actions = batch["avail_actions"]

        # Calculate estimated Q-Values
        # ============ Agent Network（Main Network）のQ値の現在の値を求める ============
        mac_out = []
        # Agent Networkの隠れ状態を初期化
        self.mac.init_hidden(batch.batch_size)

        # バッチの各タイムステップごと
        for t in range(batch.max_seq_length):
            # Agent NetworkからのQ値の出力を求める
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)

        # エージェントごとに時系列を連結
        # [ [t_1の全エージェント分の出力], [t_2], ... , [t_n] ] だったのが
        # [ [Agent1のt_1 ~ t_nの出力], [Agent2のt_1 ~ t_nの出力], ... ] となる
        # ↑（2025/07/18追記）mac_outのshapeを見ると [バッチサイズ, (最大)エピソード長, エージェント数, 行動数] となっているので違う気がする
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Pick the Q-Values for the actions taken by each agent
        # 全行動分のQ値から、実際に選択された行動のQ値を選ぶ
        # Shape: [バッチサイズ, (最大)エピソード長-1, エージェント数]
        chosen_action_qvals = th.gather(
            mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        # ============ Agent Network（Target Network）で目標値計算用のQ値を求める ============
        # 上とやることは同じ
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        # 目標値は次状態のQ値を用いるので最初のタイムステップは不要
        target_mac_out = th.stack(
            target_mac_out[1:], dim=1)  # Concat across time

        # Mask out unavailable actions
        # 選択不可能な行動はマイナスの大きい数を入れて、最大のQ値に選ばれないようにする
        if self.args.mask_unavailable_actions:
            target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        # 次状態における最大のQ値

        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            # DDQNの場合
            # --------------
            # DDQNでは目標値を見積もる際、
            # まずMain Networkを用いて次状態の最大価値の行動を求め（ここが重要）、
            # Target Networkを用いてその行動のQ値を求める
            # DQN(2016 Nature)では行動を決める時もTarget Networkを用いていた
            # --------------
            # エージェントごとのQ値の出力（予測値）
            mac_out_detach = mac_out.clone().detach()
            # 選択不可能な行動をつぶす
            if self.args.mask_unavailable_actions:
                mac_out_detach[avail_actions == 0] = -9999999
            # Main Networkで次状態における最大価値の行動を求める
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            # 上で求めた行動に対応するQ値はTarget Networkから抽出する
            # (最大価値の行動をTarget Networkを用いて選択してはだめなのだろうか？)
            target_max_qvals = th.gather(
                target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            # DDQNでない場合
            # 単純にTarget Networkの次状態の最大Q値
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        # Mixing Networkに入力して重みづけして足されたQ値を得る
        # chose_action_qvals や target_max_qvals はここまで各エージェントのQ_iだったが、ここでQ_totになりShapeも変わる
        if self.mixer is not None:
            # （現在の予測値）
            # 入力 : 全エージェントの実際に選択された行動のQ値 & グローバルな状態
            # Shape: [バッチサイズ, (最大)エピソード長, 1]
            chosen_action_qvals_sample0 = chosen_action_qvals[0].clone()
            if t_env - self.log_stats_t >= self.args.learner_log_interval:
                chosen_action_qvals = self.mixer(
                    chosen_action_qvals, batch["state"][:, :-1], wandb_log=True, t_env=t_env)
            else:
                chosen_action_qvals = self.mixer(
                    chosen_action_qvals, batch["state"][:, :-1])

            if self.args.override_wait_action_Qi_to_0:
                # 待機行動のQ_iを0に上書きする
                is_wait_action = (actions.detach().squeeze() == self.args.n_actions - 1)
                # 0代入でなく0乗算なのは計算グラフを更新するために必要らしいため
                # 将来的に0以外にしたい場合はtorch.whereを使うのが良さそう
                chosen_action_qvals = chosen_action_qvals * (~is_wait_action).float()

            # (次状態の最大Q値)
            # 入力 : 全エージェントの次状態における最大のQ値 & グローバルな次状態
            # Shape: [バッチサイズ, (最大)エピソード長, 1]
            target_max_qvals = self.target_mixer(
                target_max_qvals, batch["state"][:, 1:])

        # Calculate 1-step Q-Learning targets
        # Q値の目標値を計算
        # 目標値 = 現在の報酬 + 次状態の最大Q値を割り引いたもの
        # （終端状態は報酬のみ）
        targets = rewards + self.args.gamma * \
            (1 - terminated) * target_max_qvals

        # Td-error
        # TD誤差 = 予測値(実際に選択した行動のQ値）- 目標値
        # Shape: [バッチサイズ, (最大)エピソード長, 1]
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        # 無効なタイムステップの部分を消す
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        # MSE（平均二乗誤差）
        loss = (masked_td_error ** 2).sum() / mask.sum()

        # Optimise
        # 誤差逆伝播
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(
            self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        # 定期的にAgent, Mixing両方のTarget Networkを更新
        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        # 定期的にログをとる
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat(
                "td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals *
                                                  mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat(
                "target_mean", (targets * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env

            # fig, ax = plt.subplots()
            # data = (chosen_action_qvals * mask).squeeze().detach().numpy()
            # sns.heatmap(
            #     data,
            #     cmap="coolwarm",
            #     vmax=2.0,
            #     center=0,
            #     vmin=-2.0,
            #     ax=ax,
            #     annot=True,
            #     fmt=".2f"
            # )
            # wandb.log({"chosen_action_qvals_mixed": wandb.Image(fig)}, step=t_env)
            # plt.close(fig)  # メモリリーク防止

            # fig, ax = plt.subplots(figsize=(8, 6))
            # data = (chosen_action_qvals_sample0 * mask[0]).squeeze().detach().numpy()
            # sns.heatmap(
            #     data,
            #     cmap="coolwarm",
            #     vmax=2.0,
            #     center=0,
            #     vmin=-2.0,
            #     ax=ax,
            #     annot=True,
            #     fmt=".2f"
            # )
            # ax.set_ylabel("Timestep")
            # ax.set_xlabel("Agent")
            # wandb.log({"chosen_action_qvals_sample0": wandb.Image(fig)}, step=t_env)
            # plt.close(fig)  # メモリリーク防止

            # fig, ax = plt.subplots(figsize=(3, 6))
            # data = (chosen_action_qvals * mask)[0].detach().numpy()
            # sns.heatmap(
            #     data,
            #     cmap="coolwarm",
            #     vmax=2.0,
            #     center=0,
            #     vmin=-2.0,
            #     ax=ax,
            #     annot=True,
            #     fmt=".2f"
            # )
            # ax.set_ylabel("Timestep")
            # wandb.log({"chosen_action_qvals_sample0_mixed": wandb.Image(fig)}, step=t_env)
            # plt.close(fig)  # メモリリーク防止

    def _update_targets(self):
        """
        Agent, MixingのTarget Networkを更新 (DDQN)
        """
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(
                th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(
            th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
