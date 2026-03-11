# 令和5年度の榊原和哉の卒業研究「フードバンクにおける食料分配手法の実用化に向けた検討　ー実用化に向けた環境の拡張とその影響に関する調査ー」
# で実験結果の集計・分析に使用したコード
# 実験①と実験②では絞り込み条件が違っているので、一部コメントアウトしている

import re
import pandas as pd
from scipy.stats import t
import matplotlib.pyplot as plt
import seaborn as sns

# 目盛り調整のticker追加
import matplotlib.ticker as ticker

import json
import wandb
import datetime
import sys

api = wandb.Api()

pd.set_option("display.max_rows", 100)

def fetch_and_format():
    # print(f"{datetime.datetime.now()}: Start to get runs")
    runs = api.runs(
        path="kazuyasakakibara/FoodBank2023", # "エンティティ名/プロジェクト名"
        filters={
            "state": "finished",
            "display_name": {"$regex": r"\d+a\d+f_lc1"}, # type: lc1のrunに限定
            "config.t_max": 1000000, # （実験①） t_max=1000000のrunに限定
            # "tags": {"$nin": ["Includes_pause", "Irregular_time"]}, # （実験①） これらのタグ（実行時間の計測がうまく行っていない）を持たないrunに限定
        },
    )
    print(f"{len(runs)} runs loaded")

    processed_data = []
    print(f"{datetime.datetime.now()}: Start to process runs into DataFrame")
    for run in runs:
        # シチュエーション名から n_agents, n_foods, situation_type を抽出
        match = re.match(
            r"^(\d+)a(\d+)f_(.+)$", run.config["env_args"]["situation_name"]
        )
        match_result = match.groups()
        n_agents = int(match_result[0])
        n_foods = int(match_result[1])
        situation_type = match_result[2]

        # n_agents の値に基づく絞り込み （実験①）
        # if n_agents not in [5, 10, 15, 20]:
        #     continue

        # n_foods の値に基づく絞り込み （実験①）
        # if n_foods not in [5, 10, 15, 20, 40, 60, 80, 160, 320]:
        #     continue

        # n_agents の値に基づく絞り込み （実験②）
        if n_agents not in [5, 10, 15, 20, 25, 30, 40, 60]:
            continue

        # n_foods の値に基づく絞り込み （実験②）
        if n_foods not in [5, 10, 15, 20, 40, 60]:
            continue
        
        # episode_limit == n_foods * 5 のものに限定 （実験②）
        if run.config["env_args"]["episode_limit"] != n_foods * 5:
            continue

        if run.group is None: # 処理の高速化のため、host名をグループ名にする
            print("download")
            metadata = json.load(run.file("wandb-metadata.json").download(replace=True))
            run.group = metadata["host"]
            run.update()

        print("*", end="", flush=True)

        processed_data.append(
            {
                "Run ID": run.id,
                "Group": run.group,
                "n_agents": n_agents,
                "n_foods": n_foods,
                "t_max": run.config["t_max"],
                "use_cuda": run.config["use_cuda"],
                "runtime": run.summary["_runtime"],
                "test_total_reward": run.summary["test_total_reward"],
                "episode_limit": run.config["env_args"]["episode_limit"],
            }
        )

    print("")

    return pd.DataFrame(processed_data)


def aggregate(df: pd.DataFrame, label_for_y: str):
    agg_df = (
        df.groupby(["n_agents", "n_foods"])
        .agg({label_for_y: ["count", "mean", "sem"]})
        .reset_index()
    )

    # 信頼区間の計算
    conf_int = t.interval(
        0.95,
        agg_df[label_for_y, "count"] - 1,
        agg_df[label_for_y, "mean"],
        agg_df[label_for_y, "sem"],
    )
    agg_df[label_for_y, "ci_lower"], agg_df[label_for_y, "ci_upper"] = conf_int
    return agg_df


def main():
    is_public_mode = ("--public" in sys.argv)
    formatted_df = fetch_and_format()

    print("formatted_df\n", formatted_df)



    # windows (卒研室の2台のPC、"kitakosi101"と"kitakosi102") と私物のmac ("MBP2021.local") で分ける

    win_df = formatted_df[
        formatted_df["Group"].isin(["kitakosi101", "kitakosi102"])
        # & ~formatted_df["use_cuda"]
    ]
    # print("win_df\n", win_df)

    win101_df = formatted_df[
        formatted_df["Group"].isin(["kitakosi101"])
        # & ~formatted_df["use_cuda"]
    ]
    # print("win101_df\n", win101_df)

    win102_df = formatted_df[
        formatted_df["Group"].isin(["kitakosi102"])
        # & ~formatted_df["use_cuda"]
    ]
    # print("win102_df\n", win102_df)

    mac_df = formatted_df[
        formatted_df["Group"].isin(["MBP2021.local"])
        # & formatted_df["use_cuda"]
    ]
    # print("mac_df\n", mac_df)



    # 各グループでのruntime（実行時の実時間）について、シチュエーションごとに count, mean, sem を計算し、信頼区間も求める
    label_for_y = "runtime"    
    win_agg = aggregate(win_df, label_for_y)
    win101_agg = aggregate(win101_df, label_for_y)
    win102_agg = aggregate(win102_df, label_for_y)
    mac_agg = aggregate(mac_df, label_for_y)



    # （runの最後に記録された） "test_total_reward" が0より大きいものを学習成功とみなし、シチュエーションごとの成功率を計算する
    success_df = formatted_df[formatted_df["test_total_reward"] > 0]
    success_count_df = (
        success_df.groupby(["n_agents", "n_foods"])
        .size()
        .reset_index(name="success_count")
    )
    total_count_df = (
        formatted_df.groupby(["n_agents", "n_foods"])
        .size()
        .reset_index(name="total_count")
    )
    success_rate_df = pd.merge(
        success_count_df, total_count_df, on=["n_agents", "n_foods"], how="right"
    ).fillna(0)
    success_rate_df["success_rate"] = (
        success_rate_df["success_count"] / success_rate_df["total_count"]
    )



    # ここから結果の描画

    def plot_heatmap(ax: plt.Axes, agg_df: pd.DataFrame, values_to_plot, fmt=".0f"):
        pivot_df = agg_df.pivot(
            index="n_agents", columns="n_foods", values=values_to_plot
        )

        pivot_df = pivot_df.reindex(
            index=formatted_df["n_agents"].unique(),
            columns=formatted_df["n_foods"].unique(),
            fill_value=None,
        )
        pivot_df.sort_index(axis=0, ascending=True, inplace=True)
        pivot_df.sort_index(axis=1, ascending=True, inplace=True)

        sns.heatmap(pivot_df, ax=ax, annot=True, fmt=fmt, vmin=0, cmap="Blues",
            xticklabels=1)
        ax.invert_yaxis()  # y軸のラベルを逆向きにする

    runtime_heatmap_fig, runtime_heatmap_axes = plt.subplots(2, 3)
    (
        (win101_count_ax, win102_count_ax, mac_count_ax),
        (win101_mean_ax, win102_mean_ax, mac_mean_ax),
    ) = runtime_heatmap_axes

    # グループ別の実行数、平均実行時間のヒートマップを描画
    win101_count_ax: plt.Axes
    win102_count_ax: plt.Axes
    mac_count_ax: plt.Axes
    win101_mean_ax: plt.Axes
    win102_mean_ax: plt.Axes
    mac_mean_ax: plt.Axes

    win101_count_ax.set_title("win101_count")
    plot_heatmap(win101_count_ax, win101_agg, (label_for_y, "count"))

    win102_count_ax.set_title("win102_count")
    plot_heatmap(win102_count_ax, win102_agg, (label_for_y, "count"))

    mac_count_ax.set_title("mac_count")
    plot_heatmap(mac_count_ax, mac_agg, (label_for_y, "count"))

    win101_mean_ax.set_title("win101_mean")
    plot_heatmap(win101_mean_ax, win101_agg, (label_for_y, "mean"))

    win102_mean_ax.set_title("win102_mean")
    plot_heatmap(win102_mean_ax, win102_agg, (label_for_y, "mean"))

    mac_mean_ax.set_title("mac_mean")
    plot_heatmap(mac_mean_ax, mac_agg, (label_for_y, "mean"))

    # runtime_heatmap_fig.tight_layout()

    # 成功率をヒートマップで描画
    success_heatmap_fig, (success_heatmap_ax, total_count_ax) = plt.subplots(1, 2)

    plot_heatmap(success_heatmap_ax, success_rate_df, "success_rate", fmt=".2f")
    success_heatmap_ax.set_title("Success Rate")

    plot_heatmap(total_count_ax, total_count_df, "total_count")
    total_count_ax.set_title("Total Count")

    # ここからグラフの描画
    sns.lmplot(data=win_df, x="n_agents", y="runtime", hue="n_foods").fig.suptitle(
        "win: n_agents-runtime"
    )
    sns.lmplot(data=win_df, x="n_foods", y="runtime", hue="n_agents").fig.suptitle(
        "win: n_foods-runtime"
    )
    sns.lmplot(data=mac_df, x="n_agents", y="runtime", hue="n_foods").fig.suptitle(
        "mac: n_agents-runtime"
    )
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(5))
    sns.lmplot(data=mac_df, x="n_foods", y="runtime", hue="n_agents").fig.suptitle(
        "mac: n_foods-runtime"
    )
    plt.xlim(left=0)
    plt.ylim(bottom=0)

    sns.lmplot(data=win101_df, x="n_agents", y="runtime", hue="n_foods").fig.suptitle(
        "win101: n_agents-runtime"
    )

    sns.lmplot(data=win101_df, x="n_foods", y="runtime", hue="n_agents").fig.suptitle(
        "win101: n_foods-runtime"
    )
    sns.lmplot(data=win102_df, x="n_agents", y="runtime", hue="n_foods").fig.suptitle(
        "win102: n_agents-runtime"
    )
    sns.lmplot(data=win102_df, x="n_foods", y="runtime", hue="n_agents").fig.suptitle(
        "win102: n_foods-runtime"
    )

    if is_public_mode: # 加工して論文等に載せる時のためのモード
        for fig in plt.get_fignums():
            fig_obj = plt.figure(fig)
            for ax in fig_obj.get_axes():
                ax.set_title("")
                ax.set_xlabel('')  # x軸のラベルを空白にする
                ax.set_ylabel('')  # y軸のラベルを空白にする

    plt.show()


if __name__ == "__main__":
    main()
