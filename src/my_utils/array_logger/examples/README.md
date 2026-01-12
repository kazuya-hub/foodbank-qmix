# Examples

このディレクトリには array_logger の実用的な使用例が含まれています。

## ファイル一覧

### 1. `basic_usage.py` - 基本的な使い方
最もシンプルな使い方のデモ。初めて array_logger を使う場合はここから始めてください。

**内容:**
- ロガーの初期化・登録・記録・クローズの基本フロー
- 複数の配列タイプの登録
- データの読み取りとフィルタリング

**実行方法:**
```bash
cd examples
python basic_usage.py
```

### 2. `qmix_training.py` - 強化学習での実用例
QMIX風のマルチエージェント強化学習での使用例。実際の研究・実験に近い形での使い方を示します。

**内容:**
- 複数エージェントのQ値の記録
- ミキシングネットワークの重み保存
- エピソード報酬の記録
- TD損失の記録
- 学習結果の基本的な分析

**実行方法:**
```bash
cd examples
python qmix_training.py
```

### 3. `analysis_pipeline.py` - データ分析パイプライン
記録したデータを使った事後分析の例。学習完了後にデータを詳しく分析する際の参考になります。

**内容:**
- 学習曲線の分析
- エピソードごとの統計情報抽出
- エージェント間の比較
- チェックポイントデータの抽出
- matplotlib を使ったビジュアライゼーション（オプション）

**実行方法:**
```bash
cd examples
python analysis_pipeline.py
```

## 使い方のヒント

### 新しいプロジェクトで使う場合

1. `basic_usage.py` を参考に、記録したい配列の種類を決める
2. `register()` で各配列タイプを登録
3. 実験ループ内で `log()` を呼び出す
4. 実験終了時に `close()` を呼ぶ

### 既存コードに統合する場合

```python
import array_logger

# 実験開始時に追加
array_logger.init(root_path="./logs")
array_logger.register("my_array", keys=["episode"], shape=(10,), dtype="float32")

# 既存のループに追加
for episode in range(num_episodes):
    # ... 既存のコード ...
    
    # 記録したいタイミングで呼ぶ
    array_logger.log("my_array", {"episode": episode}, my_data)

# 実験終了時に追加
array_logger.close()
```

### パフォーマンスチューニング

commit閾値を調整することで、書き込み頻度とメモリ使用量のトレードオフを調整できます：

```python
array_logger.register(
    name="high_frequency_data",
    keys=["step"],
    shape=(100,),
    dtype="float32",
    commit_threshold_rows=1000,      # 大きくする = commit頻度↓, メモリ使用量↑
    commit_threshold_seconds=120.0   # 長くする = 時間ベースのcommit頻度↓
)
```

## トラブルシューティング

### ImportError が発生する場合
スクリプトが `sys.path` を調整して親ディレクトリから array_logger をインポートしています。
もし問題が発生する場合は、以下のように直接実行してください：

```bash
cd /path/to/array_logger
PYTHONPATH=.. python examples/basic_usage.py
```

### DBファイルが見つからない場合
スクリプトを実行したディレクトリ内に `demo_logs/`, `qmix_logs/`, `analysis_logs/` が作成されます。
絶対パスで指定したい場合は、各スクリプト内の `root_path` を変更してください。

## 次のステップ

- [README.md](../README.md) で全体的なAPI仕様を確認
- [DESIGN.md](../../docs/DESIGN.md) で内部設計を理解
- 実際のプロジェクトに統合してみる
