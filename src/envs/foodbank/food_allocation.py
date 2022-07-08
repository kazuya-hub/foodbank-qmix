import numpy as np
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


class FoodAllocationEnv():
    """
    The food allocation environment for decentralised multi-agent
    micromanagement scenarios in Food Bank.

    フードバンクにおけるマルチエージェント食品分配シミュレーション環境
    """

    def __init__(self, n_agents, n_foods, requests, initial_stock, full_observable, step_cost, max_steps, clock, debug, seed):
        self.n_agents = n_agents
        self.n_foods = n_foods
        self.requests = requests
        self.initial_stock = initial_stock
        self.max_steps = max_steps

        self._step_count = None
        self._step_cost = step_cost
        self.full_observable = full_observable
        self._add_clock = clock
        self.debug = debug
        self.episode_limit = max_steps
        self._episode_count = 0
        self.timeouts = 0

    def reset(self):
        """
        環境を初期化
        エージェントの観測を返す
        """
        # バンクの在庫をリセット
        self.bank_stock = copy.deepcopy(self.initial_stock)
        # エージェントの在庫をリセット
        self.agents_stock = np.zeros((self.n_agents, self.n_foods))

        # 要求が満たされたか
        self.agents_done = [False for _ in range(self.n_agents)]

        if self.debug:
            logging.debug(
                "Started Episode {}".format(self._episode_count).center(
                    60, "*"
                )
            )

        return self.get_obs(), self.get_state()

    def get_env_info(self):
        """
        環境のパラメータ
        """
        env_info = {
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size(),
            "n_actions": self.get_total_actions(),
            "n_agents": self.n_agents,
            "episode_limit": self.max_steps,
        }
        return env_info

    def get_obs_size(self):
        """
        Returns the size of the observation.
        - 自身の各食品の満足度 (0.0~1.0)
        - 各食品の残量 (0.0~1.0)
        """
        return 2 * self.n_foods

    def get_state_size(self):
        """
        Returns the size of the global state.
        """
        return self.get_obs_size() * self.n_agents

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take."""
        return self.n_foods + 1

    def get_avail_actions(self):
        """
        全エージェントの選択可能な行動をリストで返す
        """
        avail_actions = []
        for agent_i in range(self.n_agents):
            avail_agent = [1] * self.n_actions
            avail_actions.append(avail_agent)
        return avail_actions

    def get_obs(self):
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

            if self.debug:
                logging.debug("Obs Agent: {}".format(agent_i).center(60, "-"))
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
        obs_concat = np.concatenate(self.get_obs(), axis=0).astype(
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
