import os
import re
import cProfile
import pstats
import pprint
import traceback
import random
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.collections import QuadMesh
import matplotlib as mpl
from matplotlib.colors import ListedColormap

from src.envs.foodbank.food_situations import get_food_params


def black():
    print("\033[30m", end="")


def red():
    print("\033[31m", end="")


def green():
    print("\033[32m", end="")


def yellow():
    print("\033[33m", end="")


def blue():
    print("\033[34m", end="")


def magenta():
    print("\033[35m", end="")


def cyan():
    print("\033[36m", end="")


def white():
    print("\033[37m", end="")


def end():
    print("\033[0m", end="")


def read_log_file(file_path: str):
    with open(file_path, encoding="utf-8", mode="r") as file:
        lines = file.readlines()

    log_entries: list[str] = [""]

    for line in lines:
        # 行が"["で始まる場合、新しいエントリが始まると見なす
        if line[0] == "[":
            log_entries[-1] = log_entries[-1].rstrip("\n")
            log_entries.append(line)
        else:
            # 直前のエントリに行を追加する
            log_entries[-1] += line

    return log_entries


class LogIterater:
    def __init__(self, log_entries: list[str]) -> None:
        self.index = 0
        self.entries = log_entries.copy()
        self.buffer: list[str] = []

    def current(self):
        if len(self.entries) <= self.index:
            raise StopIteration

        return self.entries[self.index]

    def next(self):
        self.buffer.append(self.current())
        self.index += 1
        return self.current()

    def cancel(self):
        popped = self.buffer.pop()
        self.index -= 1
        return popped

    def get_buffer(self):
        """バッファの中身 (現在行の直前までの履歴) を返す"""
        return self.buffer.copy()

    def clear_buffer(self):
        self.buffer.clear()


class LogParser:
    def __init__(self, log_entries: list[str]):
        self.log_iter = LogIterater(log_entries)

        self.STDOUT_LOG = False
        self.PAUSE_FLAG = False

        try:
            while "Experiment Parameters:" not in self.log_iter.next():
                pass  # "Experiment Parameters" まで進める
            self.situation_name = re.search(
                r"'situation_name': '([^']+)'", self.log_iter.next()
            ).group(1)
            self.ex_situation = get_food_params(self.situation_name)
            print(self.situation_name)
            agents = tuple(range(self.ex_situation["n_agents"]))

            parsed_data = {
                "parameters": self.ex_situation,
                "episodes": [],
            }

            # ログの最後まで行くループ
            while True:
                try:
                    # エピソードの始まりまで行く
                    while "Started Episode " not in self.log_iter.next():
                        pass
                    episode_num = int(
                        re.search(
                            r"Started Episode (\d+)", self.log_iter.current()
                        ).group(1)
                    )
                    # エピソードの始まりに到達
                    if self.STDOUT_LOG:
                        blue()
                        print("\n".join(self.log_iter.get_buffer()))
                        end()
                        self.log_iter.clear_buffer()
                        if self.PAUSE_FLAG:
                            input()

                    steps = []
                    episode = {"episode_num": episode_num, "steps": steps}
                    parsed_data["episodes"].append(episode)

                    float_list_pattern = re.compile(r"\[((?:\s*-?(?:\d+\.\d*|inf)\s*)+)\]")
                    int_list_pattern = re.compile(r"\[([-\d\s]+)\]")

                    def parse_float_list(string, count=-1):
                        """stringがfloatの配列を表現していると仮定してパースする

                        Args:
                            string (str):
                            count (int, optional): 期待されるfloatの個数
                        """
                        list_str = float_list_pattern.search(string).group(1)

                        # fromiterを使うよりも高速だが、countが実際の個数より少ない場合に検出できない
                        result = np.fromstring(
                            list_str, sep=" ", dtype=np.float32, count=count
                        )

                        # fromstringより少し低速だがこちらならcountの間違いを検出できる
                        # splitted = list_str.split()
                        # if len(splitted) != count:
                        #     raise ValueError(
                        #         f"count discrepancy: Expected {count} but parsed {len(splitted)} items."
                        #     )
                        # result = np.fromiter(
                        #     map(float, splitted), dtype=float, count=count
                        # )

                        # print(string)
                        # print(list_str)
                        # print(result)
                        # input()
                        return result

                    def parse_int_list(string, count=-1):
                        list_str = int_list_pattern.search(string).group(1)
                        result = np.fromstring(
                            list_str, sep=" ", dtype=int, count=count
                        )
                        # print(string)
                        # print(result)
                        # input()
                        return result

                    # 最初のステップとそれ以降の内容は違うようなので分けている
                    first_step = {
                        "bank_stock": [],
                        "agent_stock": [],
                        "agent_request": [],
                        "avail_food": [],
                        "obs": [],
                        "Q_values": [],
                        "avail_actions": [],
                        "selected_actions": [],
                        "actions_succeeded": [],
                    }
                    steps.append(first_step)

                    assert "Started Episode" in self.log_iter.current()
                    assert "Bank Stock" in self.log_iter.next()
                    first_step["bank_stock"] = parse_int_list(
                        self.log_iter.next(), count=self.ex_situation["n_foods"]
                    )
                    assert "Agent Stock" in self.log_iter.next()
                    for j in agents:
                        assert f"Agent{j}: " in self.log_iter.next()
                        first_step["agent_stock"].append(
                            parse_float_list(
                                self.log_iter.current(),
                                count=self.ex_situation["n_foods"],
                            )
                        )
                    assert "Agent Request" in self.log_iter.next()
                    for j in agents:
                        assert f"Agent{j}: " in self.log_iter.next()
                        first_step["agent_request"].append(
                            parse_int_list(
                                self.log_iter.current(),
                                count=self.ex_situation["n_foods"],
                            )
                        )
                    for j in agents:
                        assert f"Agent{j} Avail Food: " in self.log_iter.next()
                        first_step["avail_food"].append(
                            parse_float_list(
                                self.log_iter.current(),
                                count=self.ex_situation["n_foods"],
                            )
                        )
                    for j in agents:
                        assert f"Obs Agent{j}" in self.log_iter.next()
                        first_step["obs"].append(
                            parse_float_list(
                                self.log_iter.next(),
                                count=self.ex_situation["n_foods"] * 2,
                            )
                        )
                    assert "Q-values" in self.log_iter.next()
                    for j in agents:
                        assert f"Agent{j}: " in self.log_iter.next()
                        first_step["Q_values"].append(
                            parse_float_list(
                                self.log_iter.current(),
                                count=self.ex_situation["n_foods"] + 1,  # +1はNo-opの分
                            )
                        )
                    assert "Available actions" in self.log_iter.next()
                    for j in agents:
                        assert f"Agent{j}: " in self.log_iter.next()
                        first_step["avail_actions"].append(
                            parse_int_list(
                                self.log_iter.current(),
                                count=self.ex_situation["n_foods"] + 1,  # +1はNo-opの分
                            )
                        )
                    for j in agents:
                        couldnt, food = re.search(
                            rf"Agent {j}: (?:(Couldn't )?Get a Food(\d+)|No-op)",
                            self.log_iter.next(),
                        ).groups()
                        if food is None:
                            first_step["selected_actions"].append(None)
                            first_step["actions_succeeded"].append(None)
                        else:
                            first_step["selected_actions"].append(int(food))
                            first_step["actions_succeeded"].append(not couldnt)

                    # TIMESTEP 1 以降
                    while True:
                        step = {
                            "bank_stock": [],
                            "agent_stock": [],
                            "reward": None,
                            "avail_food": [],
                            "obs": [],
                            "Q_values": [],
                            "avail_actions": [],
                            "selected_actions": [],
                            "actions_succeeded": [],
                            "agents_satisfaction": [],
                            "leftover_count": None,
                            "mean_satis": None,
                            "std_satis": None,
                            "reward_satisfaction": None,
                            "timeout": None,
                        }
                        steps.append(step)

                        # try:
                        ts_match = re.search(r"TIMESTEP (\d+)", self.log_iter.next())
                        if ts_match:
                            t = int(ts_match.group(1))
                        else:
                            # マッチしないのは多分前のステップで分配が完了したときだけ
                            self.log_iter.cancel()

                            assert "Agents Satisfaction: " in self.log_iter.next()
                            step["agents_satisfaction"] = parse_float_list(
                                self.log_iter.current(),
                                count=self.ex_situation["n_agents"],
                            )
                            step["leftover_count"] = int(
                                re.search(
                                    r"Leftover Count: (\d+)", self.log_iter.next()
                                ).group(1)
                            )
                            step["mean_satis"] = float(
                                re.search(
                                    r"Mean Satis\.: (\d+\.\d+)", self.log_iter.next()
                                ).group(1)
                            )
                            step["std_satis"] = float(
                                re.search(
                                    r"Std Satis\.: (\d+\.\d+)", self.log_iter.next()
                                ).group(1)
                            )
                            step["reward_satisfaction"] = float(
                                re.search(
                                    r"REWARD \(Satisfaction\): (-?\d+\.\d+)",
                                    self.log_iter.next(),
                                ).group(1)
                            )

                            # ここからは最後のステップ (分配完了ボーナスが配られるだけ)
                            assert "TIMESTEP " in self.log_iter.next()
                            assert "Actions" in self.log_iter.next()
                            assert "Bank Stock" in self.log_iter.next()
                            step["bank_stock"] = parse_int_list(
                                self.log_iter.next(),
                                count=self.ex_situation["n_foods"],
                            )
                            assert "Agent Stock" in self.log_iter.next()
                            for j in agents:
                                assert f"Agent{j}: " in self.log_iter.next()
                                step["agent_stock"].append(
                                    parse_float_list(
                                        self.log_iter.current(),
                                        count=self.ex_situation["n_foods"],
                                    )
                                )

                            if "Complete Bonus: " in self.log_iter.next():
                                self.log_iter.next()  # -> Episode Completed.
                            else:
                                if "Episode Timeouts." in self.log_iter.current():
                                    step["timeout"] = True
                                else:
                                    self.log_iter.cancel()

                            step["reward"] = float(
                                re.search(
                                    r"Reward = (-?\d+\.\d+)", self.log_iter.next()
                                ).group(1)
                            )
                            for j in agents:
                                assert f"Agent{j} Avail Food: " in self.log_iter.next()
                                step["avail_food"].append(
                                    parse_float_list(
                                        self.log_iter.current(),
                                        count=self.ex_situation["n_foods"],
                                    )
                                )
                            for j in agents:
                                assert f"Obs Agent{j}" in self.log_iter.next()
                                step["obs"].append(
                                    parse_float_list(
                                        self.log_iter.next(),
                                        count=self.ex_situation["n_foods"] * 2,
                                    )
                                )
                            assert "Q-values" in self.log_iter.next()
                            for j in agents:
                                assert f"Agent{j}: " in self.log_iter.next()
                                step["Q_values"].append(
                                    parse_float_list(
                                        self.log_iter.current(),
                                        count=self.ex_situation["n_foods"] + 1,  # +1はNo-opの分
                                    )
                                )
                            assert "Available actions" in self.log_iter.next()
                            for j in agents:
                                assert f"Agent{j}: " in self.log_iter.next()
                                step["avail_actions"].append(
                                    parse_int_list(
                                        self.log_iter.current(),
                                        count=self.ex_situation["n_foods"] + 1,  # +1はNo-opの分
                                    )
                                )
                            break

                        assert "Actions" in self.log_iter.next()
                        assert "Bank Stock" in self.log_iter.next()
                        step["bank_stock"] = parse_int_list(
                            self.log_iter.next(), count=self.ex_situation["n_foods"]
                        )
                        assert "Agent Stock" in self.log_iter.next()
                        for j in agents:
                            assert f"Agent{j}: " in self.log_iter.next()
                            step["agent_stock"].append(
                                parse_float_list(
                                    self.log_iter.current(),
                                    count=self.ex_situation["n_foods"],
                                )
                            )

                        step["reward"] = float(
                            re.search(
                                r"Reward = (-?\d+\.\d+)", self.log_iter.next()
                            ).group(1)
                        )
                        for j in agents:
                            assert f"Agent{j} Avail Food: " in self.log_iter.next()
                            step["avail_food"].append(
                                parse_float_list(
                                    self.log_iter.current(),
                                    count=self.ex_situation["n_foods"],
                                )
                            )
                        for j in agents:
                            assert f"Obs Agent{j}" in self.log_iter.next()
                            step["obs"].append(
                                parse_float_list(
                                    self.log_iter.next(),
                                    count=self.ex_situation["n_foods"] * 2,
                                )
                            )
                        assert "Q-values" in self.log_iter.next()
                        for j in agents:
                            assert f"Agent{j}: " in self.log_iter.next()
                            step["Q_values"].append(
                                parse_float_list(
                                    self.log_iter.current(),
                                    count=self.ex_situation["n_foods"] + 1,  # +1はNo-opの分
                                )
                            )
                        assert "Available actions" in self.log_iter.next()
                        for j in agents:
                            assert f"Agent{j}: " in self.log_iter.next()
                            step["avail_actions"].append(
                                parse_int_list(
                                    self.log_iter.current(),
                                    count=self.ex_situation["n_foods"] + 1,  # +1はNo-opの分
                                )
                            )

                        for j in agents:
                            couldnt, food = re.search(
                                rf"Agent {j}: (?:(Couldn't )?Get a Food(\d+)|No-op)",
                                self.log_iter.next(),
                            ).groups()
                            if food is None:
                                step["selected_actions"].append(None)
                                step["actions_succeeded"].append(None)
                            else:
                                step["selected_actions"].append(int(food))
                                step["actions_succeeded"].append(not couldnt)

                    # エピソードを抜けた
                    self.log_iter.next()

                    if self.STDOUT_LOG:
                        green()
                        print("\n".join(self.log_iter.get_buffer()))
                        end()
                        self.log_iter.clear_buffer()

                        pprint.pprint(steps)

                        if self.PAUSE_FLAG:
                            input()
                except Exception:
                    if (
                        "Updated target network" in self.log_iter.current()
                        or "t_env:" in self.log_iter.current()
                    ):
                        red()
                        print(self.log_iter.current())
                        end()
                        # 理由はよくわからないが行の途中で別の実験のログが割り込んでくることすらある
                        # 割り込まれたエピソードはそれ以上先を読むのが困難になる
                        # 発生するエラーの種類が絞れないので纏めて握りつぶしている
                        pass
        except StopIteration:  # 正常に (?) 最後まで読めたときにたどり着く
            print("StopIteration")
            # pprint.pprint(parsed_data[-1])
            # print(len(parsed_data))
            # with open("temp.py", mode="w") as f:
            #     pprint.pprint(parsed_data, stream=f)
            self.data = parsed_data
        except Exception as e:
            green()
            print(
                "\n".join(
                    repr(_ + " " * 100)[:100] + "|"
                    for _ in self.log_iter.get_buffer()[-20:]
                )
            )
            end()

            print(f"current:\n{self.log_iter.current()}")

            red()
            print(traceback.format_exc())
            end()

            raise e


class LearningViewer:
    def __init__(self, data: dict):
        np.set_printoptions(threshold=np.inf, linewidth=np.inf)
        self.parameters: dict = data["parameters"]
        self.episodes: list[dict] = data["episodes"]

        cmap = "Greens"

        # GridSpecを使用して3行2列のグリッドを作成
        self.main_fig = plt.figure(figsize=(12, 9))
        gs = GridSpec(
            3,
            3,
            width_ratios=[0.05, 1, 0.02],
            height_ratios=[0.15, 0.75, 0],
            # wspace=0.02,
            # hspace=0.2,
        )

        self.bank_stock_ax = self.main_fig.add_subplot(gs[0, 1])
        sns.heatmap(
            np.ones((1, self.parameters["n_foods"])),
            ax=self.bank_stock_ax,
            cmap="Reds",
            xticklabels=[],
            vmin=0,
            vmax=max(self.parameters["initial_stock"]),
            cbar=False,
            linewidths=1,
            linecolor="#000000",
        )

        self.actions_ax = self.main_fig.add_subplot(gs[1, 0])
        sns.heatmap(
            np.zeros((self.parameters["n_agents"], 1)),
            ax=self.actions_ax,
            vmin=0,
            vmax=2,
            cmap="Oranges",
            cbar=False,
            linewidths=1,
            linecolor="#000000",
        )
        self.actions_ax.set_title(f"actions")
        self.actions_ax.set_ylabel("agent")

        self.satisfaction_ax = self.main_fig.add_subplot(gs[1, 1])
        sns.heatmap(
            np.zeros((self.parameters["n_agents"], self.parameters["n_foods"])),
            ax=self.satisfaction_ax,
            cmap=cmap,
            yticklabels=[],
            xticklabels=5,
            vmin=0,
            vmax=1,
            cbar=False,
            linewidths=1,
            linecolor="#000000",
        )
        self.satisfaction_ax.set_xlabel("food")

        # Q値を表示するための新しいウィンドウを作成
        self.q_values_fig, self.q_values_ax = plt.subplots(figsize=(6, 4))
        self.q_values_ax.set_title("Q-values")
        self.q_values_ax.set_xlabel("action")
        self.q_values_ax.set_ylabel("agent")
        sns.heatmap(
            np.zeros((self.parameters["n_agents"], self.parameters["n_foods"] + 1)),
            ax=self.q_values_ax,
            cmap="Blues",
            annot=True,
            fmt=".3f",
            # vmin=-1,
            # vmax=1,
            cbar=False,
            linewidths=1,
            linecolor="#000000",
        )

        self.current_i_episode = 0
        self.current_step = 0

        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        self.colorbar = self.main_fig.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=self.main_fig.add_subplot(gs[1, 2]),
        )
        self.colorbar.set_label("satisfaction")

        # キーイベントの登録
        self.main_fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.q_values_fig.canvas.mpl_connect("key_press_event", self.on_key)

        # 初期フレームの表示
        self.display_frame()

        plt.show()

    # キーイベントの処理
    def on_key(self, event):
        if event.key == "right" and self.current_step + 1 < len(self.episodes[self.current_i_episode]["steps"]):
            self.current_step += 1
        elif event.key == "left" and 0 < self.current_step:
            self.current_step -= 1
        elif event.key == "down":
            self.current_i_episode += 1
            if len(self.episodes) <= self.current_i_episode:
                self.current_i_episode = 0
            # self.current_step = 0
            if len(self.episodes[self.current_i_episode]["steps"]) <= self.current_step:
                self.current_step = len(self.episodes[self.current_i_episode]["steps"]) - 1
        elif event.key == "up":
            self.current_i_episode -= 1
            if self.current_i_episode <= -1:
                self.current_i_episode = len(self.episodes) - 1
            # self.current_step = 0
            if len(self.episodes[self.current_i_episode]["steps"]) <= self.current_step:
                self.current_step = len(self.episodes[self.current_i_episode]["steps"]) - 1

        self.display_frame()

    # フレームを表示する関数
    def display_frame(self):
        episode = self.episodes[self.current_i_episode]
        episode_num = episode["episode_num"]
        agent_request = np.array(episode["steps"][0]["agent_request"])

        step = episode["steps"][self.current_step]
        bank_stock = np.array(step["bank_stock"])
        agent_stock = np.array(step["agent_stock"])

        satisfactions = agent_stock / agent_request

        q_values = np.array(step["Q_values"])
        avail_actions = np.array(step["avail_actions"])

        # self.axes[1][0].cla()
        # sns.heatmap(
        #     satisfactions, cmap="Blues", ax=self.axes[1][0], xticklabels=5, vmin=0, vmax=1, cbar=False
        # )
        self.bank_stock_ax.findobj(QuadMesh)[0].set_array(bank_stock.flatten())
        self.bank_stock_ax.set_title(
            f"bank_stock (episode {episode_num}, t={self.current_step})"
        )

        self.satisfaction_ax.findobj(QuadMesh)[0].set_array(satisfactions.flatten())
        self.satisfaction_ax.set_title(
            f"satisfactions (episode {episode_num}, t={self.current_step})"
        )

        # self.q_values_ax.findobj(QuadMesh)[0].set_array(q_values.flatten())
        self.q_values_ax.clear()
        sns.heatmap(
            q_values,
            ax=self.q_values_ax,
            cmap="Blues",
            annot=True,
            fmt=".3f",
            vmin=-1,
            vmax=2,
            cbar=False,
            linewidths=1,
            linecolor="#000000",
        )
        self.q_values_ax.set_title(
            f"Q-values (episode {episode_num}, t={self.current_step})"
        )

        # 満足度マップ上に表示されるエージェントの行動を更新
        for patch in self.satisfaction_ax.patches:
            patch.remove()
        if 1 <= self.current_step:
            previous_step = episode["steps"][self.current_step - 1]
            actions: list[int] = previous_step["selected_actions"]
            succeeded: list[int] = previous_step["actions_succeeded"]
            for a, f in enumerate(actions):
                if f is None:
                    continue
                edgecolor = "Orange" if previous_step["actions_succeeded"][a] else "Red"
                self.satisfaction_ax.add_patch(
                    plt.Rectangle((f, a), 1, 1, fill=False, edgecolor=edgecolor, lw=2)
                )
                # ハッチングを追加する場合
                # self.satisfaction_ax.add_patch(plt.Rectangle((f, a), 1, 1, fill=False, edgecolor='black', hatch='//'))

            self.actions_ax.findobj(QuadMesh)[0].set_array(
                np.array(
                    [
                        (0 if actions[a] is None else 1 if succeeded[a] is False else 2)
                        for a in range(len(actions))
                    ]
                )
            )
        else:
            self.actions_ax.findobj(QuadMesh)[0].set_array(
                np.zeros(self.parameters["n_agents"])
            )

        

        print(f"episode {episode_num}, t={self.current_step}")
        # print(bank_stock)
        # print(agent_stock)
        # print(agent_request)
        print("q_values\n", q_values)
        print("avail_actions\n", avail_actions)

        self.main_fig.tight_layout()
        self.main_fig.canvas.draw()

        self.q_values_fig.tight_layout()
        self.q_values_fig.canvas.draw()


def profile():
    # file_path = ""
    file_path = os.path.join(os.getcwd(), "./results/sacred/882/run_full.log")
    # while not os.path.exists(file_path):
    #     file_path = os.path.abspath(
    #         os.path.join(
    #             os.getcwd(), f"./results/sacred/{random.randint(1, 1392)}/run_full.log"
    #         )
    #     )
    log_entries = read_log_file(file_path)
    data = LogParser(log_entries).data
    data_viewer = LearningViewer(data)

    # for i in range(1046, 1844):
    #     file_path = os.path.abspath(
    #         os.path.join(os.getcwd(), f"./results/sacred/{i}/run_full.log")
    #     )
    #     print(file_path)
    #     if os.path.exists(file_path):
    #         print(file_path)

    #         log_entries = read_log_file(file_path)
    #         # print(log_entries[20:50])

    #         data = LogParser(log_entries).data

    #         data_viewer = LearningViewer(data)


def main():
    # cProfileを使用してプロファイリングを実行
    cProfile.run("profile()", "temp_profile_stats")

    print("=profile=======================\n")

    # pstatsモジュールを使用して結果を表示
    stats = pstats.Stats("temp_profile_stats")
    stats.sort_stats(pstats.SortKey.TIME)
    cyan()
    stats.print_stats(10)

    # log_entries = read_log_file(file_path)


if __name__ == "__main__":
    main()
