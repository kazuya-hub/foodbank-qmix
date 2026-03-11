import wandb
import numpy as np
import src.my_utils.array_logger as array_logger

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd

# ライブラリのインポート
import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

import sys
from pathlib import Path

# プロジェクトルートを sys.path に追加
project_root = Path.cwd()
sys.path.insert(0, str(project_root / "src"))
# プロジェクトルートの設定をしておかないと my_utils が import できない
import my_utils.array_logger as array_logger
from my_utils.array_logger import ArrayReader

print("✓ ライブラリインポート完了")

api = wandb.Api()

def load_array_log_df(run_id: str, array_name: str) -> pd.DataFrame:
    # ArrayReader でデータを DataFrame として読み込む
    db_path = project_root / "results" / "array_logs" / run_id / f"{array_name}.db"
    print(f"✓ 読み込み中: {db_path}")
    reader = ArrayReader(str(db_path))
    reader.open()
    df = reader.to_dataframe()
    reader.close()
    print(f"✓ 読み込み完了！")
    print(f"\nデータ形状: {df.shape}")
    print(f"\narrayの形状: {df['array'].iloc[0].shape}")
    print(f"\n最初の10行:")
    print(df.head(10))
    return df



# lossの値が極端に大きい行とそれ以外に分ける
# loss_df = load_array_log_df("stc8tzl7", "loss")
# filtered_loss_df = loss_df[loss_df['array'].apply(lambda x: x[0] > 10)]
# print("\n=== 抽出したlossが10を超える行 ===")
# print(f"抽出後のデータ数: {len(filtered_loss_df)}")
# print(filtered_loss_df)



# td_error_df = load_array_log_df("stc8tzl7", "masked_td_error")
# # shape (32, 99, 1) を (32, 20) に変換
# td_error_df['array'] = td_error_df['array'].apply(lambda x: x.squeeze()[:, :20])

# merged_df = pd.merge(filtered_loss_df, td_error_df, on='t_env', suffixes=('_loss', '_td_error'))
# # td_errorの各arrayで、absが10以上の値を含む行のマスクを作成
# merged_df["abnormal_episode_mask"] = merged_df['array_td_error'].apply(lambda x: (np.abs(x) >= 10).any())

# plt.figure(figsize=(10, 6))
# plt.hist(merged_df['array_td_error'], bins=50, edgecolor='black')
# plt.xlabel('TD Error')
# plt.ylabel('Frequency')
# plt.title('Distribution of TD Error Values')
# plt.grid(True, alpha=0.3)
# plt.show()


class HeatmapViewer:
    def __init__(self, df, array_name="array", array_col='array'):
        """
        二次元ndarrayをヒートマップで可視化するGUIビューア
        
        Parameters:
        -----------
        df : pandas.DataFrame
            データフレーム
        array_col : str
            二次元ndarrayが格納されている列名
        """
        self.df = df.reset_index(drop=True)
        self.array_name = array_name
        self.array_col = array_col
        self.current_idx = 0
        
        # matplotlibのインタラクティブモードを有効化
        plt.ion()
        
        # 図とサブプロットを作成
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.fig.canvas.manager.set_window_title(f'{self.array_name} Heatmap Viewer')
        
        # キーボードイベントを接続
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # 初回描画
        self.update_plot()
        
        # 操作説明を表示
        print("=== ヒートマップビューアの操作方法 ===")
        print("↑ キー: 前の行のndarrayを表示")
        print("↓ キー: 次の行のndarrayを表示")
        print("q キー: ビューアを閉じる")
        print("=====================================")
        
    def on_key_press(self, event):
        """キーボードイベントのハンドラ"""
        # 移動量を決定（デフォルト: 1、Shift: 10、Control: 100）
        step = 1
        if 'ctrl+' in event.key or 'control+' in event.key:
            step = 100
        elif 'shift+' in event.key:
            step = 10
        
        # 修飾キーを除去したキー名を取得
        base_key = event.key.replace('ctrl+', '').replace('control+', '').replace('shift+', '')
        
        if base_key == 'up':
            # 前の行へ移動
            if self.current_idx > 0:
                self.current_idx = max(0, self.current_idx - step)
                self.update_plot()
        elif base_key == 'down':
            # 次の行へ移動
            if self.current_idx < len(self.df) - 1:
                self.current_idx = min(len(self.df) - 1, self.current_idx + step)
                self.update_plot()
        elif base_key == 'q':
            # ビューアを閉じる
            plt.close(self.fig)
            print("ビューアを閉じました")
    
    def update_plot(self):
        """現在のインデックスのndarrayをヒートマップとして描画"""
        self.ax.clear()
        
        # 現在の行のデータを取得
        current_row = self.df.iloc[self.current_idx]
        array_data = current_row[self.array_col]
        
        # プリミティブな列の値を収集
        primitive_values = []
        for col_name, col_value in current_row.items():
            if not isinstance(col_value, np.ndarray):
                primitive_values.append(f"{col_name}={col_value}")
        
        # ndarrayを二次元配列として取得
        if isinstance(array_data, np.ndarray):
            if array_data.ndim == 1:
                # 1次元の場合は、行ベクトルとして表示
                array_data = array_data.reshape(1, -1)
            elif array_data.ndim > 2:
                # 3次元以上の場合は最初の2次元だけ表示
                array_data = array_data[0]
        
        # ヒートマップを描画
        im = self.ax.imshow(array_data, cmap='coolwarm', vmax=2.0, vmin=-2.0, aspect='auto', interpolation='nearest')
        
        # 値のノーテーションを追加
        # for i in range(array_data.shape[0]):
        #     for j in range(array_data.shape[1]):
        #         text = self.ax.text(j, i, f'{array_data[i, j]:.2f}',
        #                            ha="center", va="center", color="white", fontsize=8)
        
        # # カラーバーを追加
        # if hasattr(self, 'cbar') and self.cbar is not None:
        #     try:
        #         # カラーバーの軸を取得して削除
        #         if self.cbar.ax is not None:
        #             self.cbar.ax.remove()
        #         self.cbar = None
        #     except (AttributeError, ValueError, KeyError):
        #         self.cbar = None
        # self.cbar = self.fig.colorbar(im, ax=self.ax)
        # self.cbar.set_label('TD Error Value')
        
        # タイトルと軸ラベルを設定
        primitive_info = ", ".join(primitive_values)
        self.ax.set_title(f'{self.array_name} Heatmap (Row {self.current_idx + 1}/{len(self.df)}, {primitive_info})', 
                         fontsize=14, fontweight='bold')
        self.ax.set_xlabel('Column Index')
        self.ax.set_ylabel('Row Index')
        
        # グリッド表示
        self.ax.grid(False)
        
        # 配列の統計情報を表示
        stats_text = f'Shape: {array_data.shape} | Min: {array_data.min():.4f} | Max: {array_data.max():.4f} | Mean: {array_data.mean():.4f}'
        self.ax.text(0.5, -0.1, stats_text, transform=self.ax.transAxes, 
                    ha='center', va='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 描画を更新
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

# shape: (バッチサイズ * (最大)エピソード長, エージェント数, embed_dim)
# w1_values_df = load_array_log_df("stc8tzl7", "w1_values")
# w1_values_df['array'] = w1_values_df['array'].apply(lambda x: x.reshape(32, x.shape[0] // 32 * x.shape[1], x.shape[2])[0])
# viewer = HeatmapViewer(w1_values_df, array_name="w1_values", array_col='array')
# plt.show(block=True)

chosen_action_qvals_df = load_array_log_df("d2evn79o", "chosen_action_qvals")
viewer = HeatmapViewer(chosen_action_qvals_df, array_name="chosen_action_qvals", array_col='array')
plt.show(block=True)

# fc2_weight_df = load_array_log_df("d2evn79o", "fc2_weight")
# viewer = HeatmapViewer(fc2_weight_df, array_name="fc2_weight", array_col='array')
# # 全てのarrayの[-1][:]が一致していることを確認する
# for i in range(len(fc2_weight_df)):
#     assert np.array_equal(fc2_weight_df['array'].iloc[i][-1, :], fc2_weight_df['array'].iloc[0][-1, :])
# print("✓ 全てのarrayの最後の行が一致")
# plt.show(block=True)

# fc2_bias_df = load_array_log_df("d2evn79o", "fc2_bias")
# viewer = HeatmapViewer(fc2_bias_df, array_name="fc2_bias", array_col='array')
# # 全てのarrayの[-1]が一致していることを確認する
# for i in range(len(fc2_bias_df)):
#     assert np.array_equal(fc2_bias_df['array'].iloc[i][-1], fc2_bias_df['array'].iloc[0][-1])
# print("✓ 全てのarrayの最後の項目が一致")
# plt.show(block=True)