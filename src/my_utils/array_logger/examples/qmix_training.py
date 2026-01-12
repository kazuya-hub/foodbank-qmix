"""
強化学習での実用例
==================

QMIX や DQN などのマルチエージェント強化学習で
array_logger を使う実践的な例を示します。

記録内容:
- Q値（各エージェント）
- ミキシングネットワークの重み
- 報酬
- エピソード統計
"""

import numpy as np
import sys
from pathlib import Path
from datetime import datetime

script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir.parent.parent))
import array_logger


def simulate_qmix_training():
    """
    QMIX風のマルチエージェント学習をシミュレート
    """
    # 設定
    num_agents = 3
    num_states = 50
    num_actions = 10
    mixing_network_size = (64, 32)
    
    num_episodes = 100
    steps_per_episode = 50
    
    # 1. ロガー初期化
    print("=== QMIX Training Logger ===\n")
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    log_dir = f"./test_output/{timestamp}"
    array_logger.init(root_path=log_dir)
    
    # 2. 各種配列を登録
    print("配列を登録中...")
    
    # 各エージェントのQ値
    for agent_id in range(num_agents):
        array_logger.register(
            name=f"agent_{agent_id}_qvalues",
            keys=["episode", "step"],
            shape=(num_states, num_actions),
            dtype="float32",
            commit_threshold_rows=200
        )
    
    # ミキシングネットワークの重み（定期的に保存）
    array_logger.register(
        name="mixing_weights",
        keys=["episode"],
        shape=mixing_network_size,
        dtype="float32",
        commit_threshold_rows=10
    )
    
    # 報酬（エピソード全体の累積報酬）
    array_logger.register(
        name="episode_reward",
        keys=["episode"],
        shape=(1,),
        dtype="float32",
        commit_threshold_rows=50
    )
    
    # 損失値（各更新ステップ）
    array_logger.register(
        name="td_loss",
        keys=["episode", "step"],
        shape=(1,),
        dtype="float32",
        commit_threshold_rows=200
    )
    
    print(f"   - {num_agents} agents のQ値")
    print("   - mixing_weights")
    print("   - episode_reward")
    print("   - td_loss")
    
    # 3. 学習ループ
    print(f"\n学習開始: {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        episode_reward_sum = 0.0
        
        for step in range(steps_per_episode):
            # 各エージェントのQ値を記録
            for agent_id in range(num_agents):
                q_values = np.random.rand(num_states, num_actions).astype(np.float32)
                array_logger.log(
                    name=f"agent_{agent_id}_qvalues",
                    key_values={"episode": episode, "step": step},
                    array=q_values
                )
            
            # TD損失を記録
            td_loss = np.random.rand(1).astype(np.float32)
            array_logger.log(
                name="td_loss",
                key_values={"episode": episode, "step": step},
                array=td_loss
            )
            
            # 報酬の累積
            episode_reward_sum += np.random.rand() * 10
        
        # エピソード終了時の報酬を記録
        array_logger.log(
            name="episode_reward",
            key_values={"episode": episode},
            array=np.array([episode_reward_sum], dtype=np.float32)
        )
        
        # 10エピソードごとにミキシングネットワークの重みを保存
        if episode % 10 == 0:
            mixing_weights = np.random.rand(*mixing_network_size).astype(np.float32)
            array_logger.log(
                name="mixing_weights",
                key_values={"episode": episode},
                array=mixing_weights
            )
            
            print(f"   Episode {episode}: reward={episode_reward_sum:.2f}")
    
    # 4. クローズ
    print("\n学習完了、ロガーをクローズ...")
    array_logger.close()
    print("✓ すべてのデータをcommit完了")
    
    return log_dir


def analyze_training_results(log_dir: str):
    """
    記録したデータを分析
    """
    print("\n=== 学習結果の分析 ===\n")
    
    # 報酬の推移を分析
    print("1. 報酬の推移:")
    reader = array_logger.open_reader(f"{log_dir}/episode_reward.db")
    
    rewards = []
    for keys_dict, array in reader.iterate():
        rewards.append(array[0])
    
    # 統計情報
    rewards_np = np.array(rewards)
    print(f"   - 平均報酬: {rewards_np.mean():.2f}")
    print(f"   - 最大報酬: {rewards_np.max():.2f}")
    print(f"   - 最小報酬: {rewards_np.min():.2f}")
    print(f"   - 標準偏差: {rewards_np.std():.2f}")
    
    # 初期10エピソード vs 最後10エピソード
    print(f"   - 初期10エピソード平均: {rewards_np[:10].mean():.2f}")
    print(f"   - 最後10エピソード平均: {rewards_np[-10:].mean():.2f}")
    
    reader.close()
    
    # 特定エピソードのQ値を分析
    print("\n2. Agent 0 のQ値分析（Episode 50）:")
    reader = array_logger.open_reader(f"{log_dir}/agent_0_qvalues.db")
    
    q_values_list = []
    for keys_dict, array in reader.iterate(where_clause="episode = ?", params=(50,)):
        q_values_list.append(array)
    
    if q_values_list:
        q_values_ep50 = np.stack(q_values_list)  # shape: (steps, states, actions)
        print(f"   - 取得ステップ数: {len(q_values_list)}")
        print(f"   - Q値の平均: {q_values_ep50.mean():.4f}")
        print(f"   - Q値の最大: {q_values_ep50.max():.4f}")
        print(f"   - Q値の最小: {q_values_ep50.min():.4f}")
    
    reader.close()
    
    # ミキシングネットワークの重み履歴
    print("\n3. ミキシングネットワークの重み保存履歴:")
    reader = array_logger.open_reader(f"{log_dir}/mixing_weights.db")
    
    saved_episodes = []
    for keys_dict, array in reader.iterate():
        saved_episodes.append(keys_dict['episode'])
    
    print(f"   - 保存回数: {len(saved_episodes)}")
    print(f"   - 保存されたエピソード: {saved_episodes}")
    
    reader.close()
    
    print("\n✓ 分析完了")


def main():
    # 学習実行
    log_dir = simulate_qmix_training()
    
    # 結果分析
    analyze_training_results(log_dir)
    
    print("\n=== すべて完了 ===")
    print(f"\nDBファイルの場所: {log_dir}/")


if __name__ == "__main__":
    main()
