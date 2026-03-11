"""
基本的な使い方のデモ
=====================

このスクリプトは array_logger の最も基本的な使い方を示します。
実際の強化学習実験では、このパターンをベースに拡張できます。
"""

import numpy as np
import sys
from pathlib import Path
from datetime import datetime

# array_loggerをインポート（親ディレクトリから）
script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir.parent.parent))
import array_logger


def main():
    print("=== array_logger 基本デモ ===\n")
    
    # 1. ロガーの初期化
    # ────────────────────
    # 実験開始時に1回だけ呼ぶ
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    log_dir = f"./test_output/{timestamp}"
    print(f"1. ロガーを初期化: {log_dir}")
    array_logger.init(root_path=log_dir)
    
    # 2. 配列の登録
    # ────────────────────
    # 記録したい配列の種類を事前に登録
    # 各配列タイプごとに独立したDBファイルが作成される
    
    print("\n2. 配列を登録:")
    
    # 例1: Q値（状態-行動価値）
    array_logger.register(
        name="q_values",
        keys=["episode", "step"],      # インデックス用のキー
        shape=(10, 5),                 # (状態数, 行動数) の形状
        dtype="float32",
        commit_threshold_rows=50       # 50行ごとに自動commit
    )
    print("   - q_values: shape=(10, 5), keys=['episode', 'step']")
    
    # 例2: 損失値（スカラー値を1次元配列として保存）
    array_logger.register(
        name="loss",
        keys=["episode"],
        shape=(1,),                    # スカラー値も配列として扱う
        dtype="float32",
        commit_threshold_rows=100
    )
    print("   - loss: shape=(1,), keys=['episode']")
    
    # 3. データの記録
    # ────────────────────
    # 実験ループ内で繰り返し呼ぶ
    
    print("\n3. データを記録中...")
    num_episodes = 10
    steps_per_episode = 20
    
    for episode in range(num_episodes):
        # エピソードごとの損失を記録
        loss_value = np.random.rand(1).astype(np.float32)
        array_logger.log(
            name="loss",
            key_values={"episode": episode},
            array=loss_value
        )
        
        # ステップごとのQ値を記録
        for step in range(steps_per_episode):
            # ダミーのQ値（実際には学習モデルから取得）
            q = np.random.rand(10, 5).astype(np.float32)
            
            array_logger.log(
                name="q_values",
                key_values={"episode": episode, "step": step},
                array=q
            )
        
        if episode % 3 == 0:
            print(f"   Episode {episode}: loss={loss_value[0]:.4f}")
    
    print(f"   → 合計 {num_episodes} episodes, {num_episodes * steps_per_episode} steps 記録完了")
    
    # 4. ロガーのクローズ
    # ────────────────────
    # 実験終了時に必ず呼ぶ（未commitのデータをフラッシュ）
    
    print("\n4. ロガーを終了（未commitデータをフラッシュ）")
    array_logger.close()
    
    # 5. データの読み取り
    # ────────────────────
    # 記録したデータを分析
    
    print("\n5. データを読み取り:")
    
    # Q値を読む
    print("\n   [Q値の読み取り]")
    reader = array_logger.open_reader(f"{log_dir}/q_values.db")
    
    # 特定のエピソードのみフィルタ
    count = 0
    for keys_dict, array in reader.iterate(where_clause="episode = ?", params=(5,)):
        if count < 3:  # 最初の3件のみ表示
            print(f"   Episode {keys_dict['episode']}, Step {keys_dict['step']}: "
                  f"shape={array.shape}, dtype={array.dtype}, "
                  f"mean={array.mean():.4f}")
        count += 1
    
    print(f"   → Episode 5 で合計 {count} ステップ分のデータを取得")
    reader.close()
    
    # 損失を読む
    print("\n   [損失の読み取り]")
    reader = array_logger.open_reader(f"{log_dir}/loss.db")
    
    losses = []
    for keys_dict, array in reader.iterate():
        losses.append((keys_dict['episode'], array[0]))
    
    # 最初と最後のいくつかを表示
    print("   最初の3エピソード:")
    for ep, loss in losses[:3]:
        print(f"      Episode {ep}: loss={loss:.4f}")
    
    print("   最後の3エピソード:")
    for ep, loss in losses[-3:]:
        print(f"      Episode {ep}: loss={loss:.4f}")
    
    reader.close()
    
    print("\n=== デモ完了 ===")
    print(f"\nDBファイルの場所: {log_dir}/")
    print("   - q_values.db")
    print("   - loss.db")


if __name__ == "__main__":
    main()
