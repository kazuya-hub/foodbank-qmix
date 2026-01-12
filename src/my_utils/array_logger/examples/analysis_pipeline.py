"""
データ分析パイプライン例
========================

array_logger で記録したデータを使って、
学習後の分析やビジュアライゼーションを行う例。

matplotlib を使ったグラフ描画も含みます。
"""

import numpy as np
import sys
from pathlib import Path
from datetime import datetime

script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir.parent.parent))
import array_logger


def analyze_learning_curve(log_dir: str):
    """
    学習曲線を分析・可視化
    """
    print("=== 学習曲線の分析 ===\n")
    
    # 損失の推移を読み込み
    print("\n1. TD損失の推移を分析:")
    reader = array_logger.open_reader(f"{log_dir}/td_loss.db")
    
    episodes = []
    steps = []
    losses = []
    
    for keys_dict, array in reader.iterate():
        episodes.append(keys_dict['episode'])
        steps.append(keys_dict['step'])
        losses.append(array[0])
    
    reader.close()
    
    # エピソードごとの平均損失を計算
    episodes_np = np.array(episodes)
    losses_np = np.array(losses)
    
    unique_episodes = np.unique(episodes_np)
    avg_losses = []
    
    for ep in unique_episodes:
        mask = episodes_np == ep
        avg_loss = losses_np[mask].mean()
        avg_losses.append(avg_loss)
    
    print(f"   - 総エピソード数: {len(unique_episodes)}")
    print(f"   - 初期10エピソード平均損失: {np.mean(avg_losses[:10]):.4f}")
    print(f"   - 最後10エピソード平均損失: {np.mean(avg_losses[-10:]):.4f}")
    
    # matplotlib があればグラフ描画
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.plot(unique_episodes, avg_losses, label='Average TD Loss')
        
        # 移動平均も表示
        window = 10
        if len(avg_losses) >= window:
            moving_avg = np.convolve(avg_losses, np.ones(window)/window, mode='valid')
            plt.plot(unique_episodes[window-1:], moving_avg, 
                    label=f'{window}-episode Moving Average', linewidth=2)
        
        plt.xlabel('Episode')
        plt.ylabel('TD Loss')
        plt.title('Learning Curve: TD Loss over Episodes')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{log_dir}/learning_curve.png', dpi=150)
        print(f"\n   ✓ グラフを保存: {log_dir}/learning_curve.png")
        
    except ImportError:
        print("\n   (matplotlib がインストールされていないため、グラフはスキップ)")


def compare_agents_performance(log_dir: str):
    """
    複数エージェントのQ値を比較
    """
    print("\n=== エージェント間のQ値比較 ===\n")
    
    num_agents = 3
    target_episode = 50
    
    agent_stats = []
    
    for agent_id in range(num_agents):
        reader = array_logger.open_reader(f"{log_dir}/agent_{agent_id}_qvalues.db")
        
        q_values_list = []
        for keys_dict, array in reader.iterate(
            where_clause="episode = ?", 
            params=(target_episode,)
        ):
            q_values_list.append(array)
        
        reader.close()
        
        if q_values_list:
            q_values = np.stack(q_values_list)
            stats = {
                'agent_id': agent_id,
                'mean': q_values.mean(),
                'std': q_values.std(),
                'max': q_values.max(),
                'min': q_values.min()
            }
            agent_stats.append(stats)
            
            print(f"Agent {agent_id}:")
            print(f"   - 平均Q値: {stats['mean']:.4f}")
            print(f"   - 標準偏差: {stats['std']:.4f}")
            print(f"   - 最大Q値: {stats['max']:.4f}")
            print(f"   - 最小Q値: {stats['min']:.4f}\n")


def extract_checkpoint_data(log_dir: str):
    """
    特定エピソードのチェックポイントデータを抽出
    """
    print("=== チェックポイントデータの抽出 ===\n")
    
    checkpoint_episodes = [0, 25, 50, 75, 99]
    
    print(f"抽出対象エピソード: {checkpoint_episodes}\n")
    
    for ep in checkpoint_episodes:
        # Q値の統計
        reader = array_logger.open_reader(f"{log_dir}/agent_0_qvalues.db")
        
        count = 0
        q_sum = 0.0
        
        for keys_dict, array in reader.iterate(
            where_clause="episode = ?",
            params=(ep,)
        ):
            count += 1
            q_sum += array.mean()
        
        reader.close()
        
        if count > 0:
            avg_q = q_sum / count
            print(f"Episode {ep:3d}: 平均Q値 = {avg_q:.4f} ({count} steps)")


def _generate_sample_data():
    """
    分析用のサンプルデータを生成
    """
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    log_dir = f"./test_output/{timestamp}"
    array_logger.init(root_path=log_dir)
    
    num_agents = 3
    num_episodes = 100
    steps_per_episode = 50
    
    # 配列を登録
    for agent_id in range(num_agents):
        array_logger.register(
            name=f"agent_{agent_id}_qvalues",
            keys=["episode", "step"],
            shape=(50, 10),
            dtype="float32",
            commit_threshold_rows=200
        )
    
    array_logger.register(
        name="td_loss",
        keys=["episode", "step"],
        shape=(1,),
        dtype="float32",
        commit_threshold_rows=200
    )
    
    # データを生成（学習の進行をシミュレート）
    for episode in range(num_episodes):
        # 学習が進むにつれて損失が減少するパターン
        base_loss = 1.0 / (1.0 + episode * 0.01)
        
        for step in range(steps_per_episode):
            # 各エージェントのQ値
            for agent_id in range(num_agents):
                # 学習が進むにつれてQ値が増加
                q_base = 0.1 + episode * 0.01 + agent_id * 0.05
                q_values = np.random.randn(50, 10).astype(np.float32) * 0.1 + q_base
                
                array_logger.log(
                    name=f"agent_{agent_id}_qvalues",
                    key_values={"episode": episode, "step": step},
                    array=q_values
                )
            
            # TD損失
            loss = base_loss + np.random.rand() * 0.1
            array_logger.log(
                name="td_loss",
                key_values={"episode": episode, "step": step},
                array=np.array([loss], dtype=np.float32)
            )
    
    array_logger.close()
    return log_dir


def main():
    print("=== array_logger データ分析パイプライン ===\n")
    
    # サンプルデータ生成
    print("サンプルデータを生成中...")
    log_dir = _generate_sample_data()
    print(f"✓ データ生成完了: {log_dir}\n")
    
    # 各種分析を実行
    analyze_learning_curve(log_dir)
    compare_agents_performance(log_dir)
    extract_checkpoint_data(log_dir)
    
    print("\n=== 分析完了 ===")


if __name__ == "__main__":
    main()
