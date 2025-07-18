# from envs import REGISTRY as env_REGISTRY
from functools import partial

import wandb
from components.episode_buffer import EpisodeBatch
import numpy as np
from controllers.basic_controller import BasicMAC

# from smac.env import StarCraft2Env
# from envs.checkers import Checkers
from envs.foodbank.food_allocation import FoodAllocationEnv
from utils.logging import Logger


class EpisodeRunner:
    """
    環境から情報を取得 & 行動を出力しながらエピソードを実行する
    """

    def __init__(self, args, logger: Logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        # 環境
        self.env = FoodAllocationEnv(logger=self.logger, **self.args.env_args)

        self.episode = 0

        # 最大タイムステップ数
        self.episode_limit = self.env.episode_limit
        # 現在のタイムステップ
        self.t = 0
        # 累計タイムステップ数
        self.t_env = 0

        # ログ用
        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac: BasicMAC):
        """
        MACを設定 & EpisodeBatchの設定
        """
        # EpisodeBatchの引数を固定したものをここで作っておく（初期化のたびに再利用するため）
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)

        # 渡されたMACを保持
        self.mac = mac

    def get_env_info(self):
        """
        環境の情報を取得（辞書型）
        """
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self, episode, test_mode=False, print_log=False):
        """
        エピソードの最初に環境など諸々を初期化
        """
        # 新しいバッチを用意
        self.batch = self.new_batch()
        # 環境をリセット
        self.env.reset(episode, test_mode=test_mode, print_log=print_log)
        # タイムステップを0に
        self.t = 0

    def run(self, episode, test_mode=False, print_log=False):
        """
        1エピソードを実行してバッチを返す
        """

        self.episode = episode

        # 環境を初期化
        self.reset(episode=episode, test_mode=test_mode, print_log=print_log)

        terminated = False  # a
        total_reward = 0  # エピソードで得られた報酬の総和

        # 隠れ状態を初期化
        self.mac.init_hidden(batch_size=self.batch_size)

        # ---------------- エピソード開始！！！ ----------------

        while not terminated:

            # 遷移前の情報を環境から取得
            pre_transition_data = {
                # グローバル状態
                "state": [self.env.get_state()],
                # 各エージェントの選択可能な行動
                "avail_actions": [self.env.get_avail_actions()],
                # エージェントの部分観測
                "obs": [self.env.get_obs()]
            }

            # バッチに遷移前の情報を追加
            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            # 現時点のバッチ（エピソードの最初から今までの遷移情報が含まれている）を渡して、Agent Networkから行動を決定
            actions = self.mac.select_actions(
                self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode, logger=self.logger, print_log=print_log)

            # 行動を出力して、環境からフィードバックを得る
            # 返り値 = 報酬，エピソードが終了したか，環境情報
            reward, terminated, env_info = self.env.step(actions[0])

            # このエピソードの総収益
            total_reward += reward

            # 遷移後の情報
            post_transition_data = {
                # 選択した行動
                "actions": actions,
                # 獲得した報酬
                "reward": [(reward,)],
                # エピソード終了の原因が目的達成による時のみTrue（最大回数を超えて終了した場合はFalse）
                "terminated": [(terminated != env_info.get("timeout", False),)],
            }

            # 遷移後の情報もバッチに追加
            self.batch.update(post_transition_data, ts=self.t)

            # タイムステップを進める
            self.t += 1

        # ---------------- エピソード終了 ----------------

        # 終端状態における情報を取得
        last_data = {
            # 終端のグローバル状態
            "state": [self.env.get_state()],
            # 選択可能な行動
            "avail_actions": [self.env.get_avail_actions()],
            # 部分観測
            "obs": [self.env.get_obs()]
        }

        # バッチに終端状態の情報を追加
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        # 終端状態における行動をAgent Networkから決定（？）
        actions = self.mac.select_actions(
            self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode, logger=self.logger, print_log=print_log)
        # バッチに最後の行動を追加
        self.batch.update({"actions": actions}, ts=self.t)

        # ------ ログをまとめる ------

        # グローバルから参照渡しで読み込む
        sum_stats = self.test_stats if test_mode else self.train_stats
        total_reward_history = self.test_returns if test_mode else self.train_returns

        # ファイル名用
        log_prefix = "test_" if test_mode else ""

        # sum_statsにenv_infoの値を加算していく
        sum_stats.update({k: sum_stats.get(k, 0) + env_info.get(k, 0)
                          for k in set(sum_stats) | set(env_info)})

        # 何エピソード分の和かわかるようにn_episodesを加算していく
        sum_stats["n_episodes"] = 1 + sum_stats.get("n_episodes", 0)
        sum_stats["episode_length"] = self.t + \
            sum_stats.get("episode_length", 0)

        # 累計タイムステップを蓄積（テスト時以外）
        if not test_mode:
            self.t_env += self.t

        # エピソード総収益の履歴を更新
        total_reward_history.append(total_reward)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            # テストの際のログ
            self._log(total_reward_history, sum_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            # 定期的にログをとる
            self._log(total_reward_history, sum_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat(
                    "epsilon", self.mac.action_selector.epsilon, self.t_env)
                wandb.log(
                    {"epsilon": self.mac.action_selector.epsilon}, step=self.t_env)
            self.log_train_stats_t = self.t_env

        # バッチを返す
        return self.batch

    def _log(self, total_reward_history: list, sum_stats: dict, prefix):
        """
        前回記録時からエピソードごとの報酬総和の平均・分散を記録
        """
        self.logger.log_stat("episode", self.episode, self.t_env)
        wandb.log({"episode": self.episode}, step=self.t_env)

        # 報酬の総和の平均・分散
        total_reward_mean = np.mean(total_reward_history)
        self.logger.log_stat(prefix + "total_reward",
                             total_reward_mean, self.t_env)
        self.logger.log_stat(prefix + "return_std",
                             np.std(total_reward_history), self.t_env)

        wandb.log({prefix + "total_reward": total_reward_mean}, step=self.t_env)

        # 総収益の履歴をクリア
        total_reward_history.clear()

        # env_infoの値の総和を記録したエピソード数で割る
        for key, value in sum_stats.items():
            if key != "n_episodes":
                stat = value / sum_stats["n_episodes"]
                self.logger.log_stat(prefix + key + "_mean",
                                     stat, self.t_env)
                wandb.log({prefix + key: stat}, step=self.t_env)

        # env_infoの履歴をクリア
        sum_stats.clear()
