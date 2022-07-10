import numpy as np
import enum
import copy
import logging

AGENTS_COUNT = 1
FOODS = [5, 5]
NUM_FOODS = len(FOODS)
REQUESTS = [
    [1, 3],
]

# AGENTS_COUNT = 2
# FOODS = [20, 20, 20]
# NUM_FOODS = len(FOODS)
# REQUESTS = [
#     [10, 10, 10],
#     [5, 10, 5],
#     [5, 5, 10],
# ]


class EpisodeStatus(enum.IntEnum):
    ONGOING = 0
    COMPLETED = 1
    TIMEOUT = 2


class FoodAllocationEnv():
    """
    The food allocation environment for decentralised multi-agent
    micromanagement scenarios in Food Bank.

    フードバンクにおけるマルチエージェント食品分配シミュレーション環境
    """

    def __init__(self, n_agents, n_foods, requests, initial_stock, full_observable, step_cost, episode_limit, clock, debug, seed):
        self.n_agents = n_agents
        self.n_foods = n_foods
        self.requests = requests
        self.initial_stock = initial_stock
        self._step_cost = step_cost
        self.episode_limit = episode_limit

        self.n_actions = self.n_foods + 1

        self._step_count = None
        self._episode_count = 0

        self.full_observable = full_observable
        self._add_clock = clock
        self.debug = debug

        self.timeouts = 0

    def reset(self):
        """
        環境を初期化
        エージェントの観測とグローバル状態を返す
        """
        # タイムステップをリセット
        self._step_count = 0
        # バンクの在庫をリセット
        self.bank_stock = copy.deepcopy(self.initial_stock)
        # エージェントの在庫をリセット
        self.agents_stock = np.zeros((self.n_agents, self.n_foods))

        # 終了フラグをリセット
        self.agents_done = [False for _ in range(self.n_agents)]

        # 最小残り個数を計算する
        # 食品ごとの要求の合計
        # requests_sum = np.sum(np.array(self.requests), axis=0)
        # 在庫 - 要求 （0以上の部分だけ足し合わせる）
        # leftover = self.initial_stock - requests_sum
        # self.ideal_min_leftover = np.sum((leftover > 0) * leftover)

        if self.debug:
            logging.debug(
                "Started Episode {}".format(self._episode_count).center(
                    60, "*"
                )
            )
            logging.debug("Bank Stock".center(60, "-"))
            logging.debug(self.bank_stock)
            logging.debug("Agent Stock".center(60, "-"))
            for agent_i in range(self.n_agents):
                logging.debug("Agent{}: {}".format(
                    agent_i, self.agents_stock[agent_i]))
            logging.debug("Agent Request".center(60, "-"))
            for agent_i in range(self.n_agents):
                logging.debug("Agent{}: {}".format(
                    agent_i, self.requests[agent_i]))

        return self.get_obs(debug=False), self.get_state()

    def step(self, agents_action):
        """
        行動を環境に出力してタイムステップを1つ進める

        -> [ 報酬, 終了フラグ, 追加情報(残り個数など) ]
        """

        # 人数分の行動が入力されているかチェック
        assert len(agents_action) == self.n_agents
        # タイムステップを進める
        self._step_count += 1

        if self.debug:
            logging.debug("Actions".center(60, "-"))

        # エージェントごと
        for agent_i, action in enumerate(agents_action):
            # 行動を出力
            self.take_action(agent_i, action)

        logging.debug("Bank Stock".center(60, "-"))
        logging.debug(self.bank_stock)
        logging.debug("Agent Stock".center(60, "-"))
        for agent_i in range(self.n_agents):
            logging.debug("Agent{}: {}".format(
                agent_i, self.agents_stock[agent_i]))

        status = self.check_status()

        reward = self.get_reward(status)

        terminated = False
        reward += self._step_cost
        info = {}

        if status is EpisodeStatus.COMPLETED:
            terminated = True
            reward += 10
            info["completed"] = True
            logging.debug("Episode Completed.")

        elif status is EpisodeStatus.TIMEOUT:
            terminated = True
            info["timeout"] = True
            self.timeouts += 1
            logging.debug("Episode Timeouts.")

        if self.debug:
            logging.debug("Reward = {}".format(reward).center(60, "-"))

        if terminated:
            self._episode_count += 1
            info["leftover"] = sum(self.bank_stock)

        return reward, terminated, info

    def take_action(self, agent_i, action):
        """
        エージェントが行動をとる（選んだ食品を獲得）
        """
        if action == self.get_total_actions() - 1:
            # No-op（何もしない）
            logging.debug("Agent {}: No-op".format(agent_i))
            return

        food = action

        if self.bank_stock[food] > 0:
            # フードバンクから１つ取る
            self.bank_stock[food] -= 1
            # 自身の在庫が1つ増える
            self.agents_stock[agent_i][food] += 1
            logging.debug(
                "Agent {}: Get a Food{}".format(agent_i, food))
        else:
            # 在庫がない（他のエージェントにもうとられた）
            # TODO: 選択した行動と一致していないので検討が必要
            logging.debug(
                "Agent {}: Couldn't Get a Food{}".format(agent_i, food))

    def get_reward(self, status):
        """
        報酬関数
        """

        reward = 0

        # エピソード終了時に報酬を与える
        if status in (EpisodeStatus.COMPLETED, EpisodeStatus.TIMEOUT):
            # 残り個数が 最小残り個数+5個以下 だった場合に報酬
            # reward_no_food_waste = 0
            # if sum(self.bank_stock) <= self.ideal_min_leftover + 5:
            #     reward_no_food_waste = 10

            # 満足度による報酬
            # 各食品の満足度 = 獲得した個数 / 要求個数
            rates = np.array(self.agents_stock) / np.array(self.requests)
            # 最大値を1に
            rates[rates > 1.0] = 1.0
            # 各エージェントの満足度 = 各食品の満足度の和
            agents_satisfaction = np.mean(rates, axis=1)

            # 平均
            reward_mean_satis = np.mean(agents_satisfaction * 100)
            # 分散
            reward_variance_satis = np.var(agents_satisfaction * 100)

            reward = reward_mean_satis + reward_variance_satis

            # logging.debug("Agents Stock: {}".format(self.agents_stock))
            logging.debug("Agents Satisfaction: {}".format(
                agents_satisfaction))
            logging.debug("Leftover Count: {}".format(sum(self.bank_stock)))

            logging.debug("REWARD (Mean Satis.): {}".format(reward_mean_satis))
            logging.debug("REWARD (Variance Satis.): {}".format(
                reward_variance_satis))

        return reward

    def check_status(self):
        """
        食品分配が終わったかどうかの状況チェック
        """
        # 最大ステップ数を超えた
        if self._step_count >= self.episode_limit:
            return EpisodeStatus.TIMEOUT

        # 獲得個数と要求個数の差
        gap = self.requests - self.agents_stock

        # 全てのエージェントの要求が満たされた
        if np.all(gap <= 0):
            return EpisodeStatus.COMPLETED

        # 要求のある食品の在庫がもうどれもない（全エージェントの取れる行動がない）
        required_count = np.sum(gap, axis=0)
        required_food_stock = np.array(self.bank_stock)[required_count > 0]
        if np.all(required_food_stock == 0):
            return EpisodeStatus.COMPLETED

        # 継続
        return EpisodeStatus.ONGOING

    def get_env_info(self):
        """
        環境のパラメータ
        """
        env_info = {
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size(),
            "n_actions": self.get_total_actions(),
            "n_agents": self.n_agents,
            "episode_limit": self.episode_limit,
        }
        return env_info

    def get_obs_size(self):
        """
        部分観測のサイズを返す
        - 自身の各食品の満足度 (0.0~1.0)
        - 各食品の残量 (0.0~1.0)
        """
        return 2 * self.n_foods

    def get_state_size(self):
        """
        グローバル状態のサイズを返す
        """
        return self.get_obs_size() * self.n_agents

    def get_total_actions(self):
        """
        エージェントがとることのできる行動の数を返す
        """
        return self.n_actions

    def get_avail_actions(self):
        """
        全エージェントの選択可能な行動をリストで返す
        """
        avail_actions = []
        for agent_i in range(self.n_agents):
            avail_agent = [1] * self.n_actions
            avail_actions.append(avail_agent)
        return avail_actions

    def get_obs(self, debug=True):
        """
        全てのエージェントの観測を1つのリストで返す
        - 各食品の残量 (0.0~1.0)
        - 自身の各食品の満足度 (0.0~1.0)
        NOTE: 分散実行時はエージェントは自分自身の観測のみ用いるようにする
        """
        _obs = []

        # 在庫残りの割合
        remaining = [0 for _ in range(self.n_foods)]
        for food in range(self.n_foods):
            # 残量率
            remaining[food] = self.bank_stock[food] / \
                self.initial_stock[food]

        # 要求が満たされた割合
        for agent_i in range(self.n_agents):
            satisfaction = [0 for _ in range(self.n_foods)]
            for food in range(self.n_foods):
                # 満足度
                satisfaction[food] = self.agents_stock[agent_i][food] / \
                    self.requests[agent_i][food]

            agent_obs = np.concatenate([remaining, satisfaction])
            _obs.append(agent_obs)

            if self.debug and debug:
                logging.debug("Obs Agent{}".format(agent_i).center(60, "-"))
                # logging.debug(
                #     "Avail. actions {}".format(
                #         self.get_avail_agent_actions(agent_id)
                #     )
                # )
                # logging.debug("Move feats {}".format(move_feats))
                # logging.debug("Enemy feats {}".format(enemy_feats))
                # logging.debug("Ally feats {}".format(ally_feats))
                # logging.debug("Own feats {}".format(own_feats))
                logging.debug(agent_obs)

        if self.full_observable:
            _obs = np.array(_obs).flatten().tolist()
            _obs = [_obs for _ in range(self.n_agents)]
        return _obs

    def get_state(self):
        """
        グローバル状態を返す
        NOTE: この関数は分散実行時は用いないこと
        """
        # 各エージェントの観測を結合したものをグローバル状態とする
        obs_concat = np.concatenate(self.get_obs(debug=False), axis=0).astype(
            np.float32
        )
        return obs_concat

        # if self.obs_instead_of_state:
        #     obs_concat = np.concatenate(self.get_obs(), axis=0).astype(
        #         np.float32
        #     )
        #     return obs_concat

        # state_dict = self.get_state_dict()

        # state = np.append(
        #     state_dict["allies"].flatten(), state_dict["enemies"].flatten()
        # )
        # if "last_action" in state_dict:
        #     state = np.append(state, state_dict["last_action"].flatten())
        # if "timestep" in state_dict:
        #     state = np.append(state, state_dict["timestep"])

        # state = state.astype(dtype=np.float32)

        # if self.debug:
        #     logging.debug("STATE".center(60, "-"))
        #     logging.debug("Ally state {}".format(state_dict["allies"]))
        #     logging.debug("Enemy state {}".format(state_dict["enemies"]))
        #     if self.state_last_action:
        #         logging.debug("Last actions {}".format(self.last_action))

        # return state
