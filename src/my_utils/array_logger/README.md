# array_logger

強化学習などの実験で発生する numpy ndarray データを効率的に永続化するためのロガーライブラリ。

## 特徴

- **SQLiteを使用**: 
- **高速・低オーバーヘッド**: WALモードを活用し、commit頻度を自動調整
- **配列単位の管理**: 配列の種類ごとに独立したDBファイルで管理
- **自己記述的**: DBファイル内に復元のためのメタデータを含むので、単体でshape/dtypeを復元可能
- **zstd圧縮サポート**: データを高い圧縮率で保存してストレージ使用量を削減（オプション）
- **シンプルなAPI**: init/register/log/closeの4つの関数のみ

## クイックスタート

```python
import numpy as np
import array_logger

# 1. 初期化（実験中に一度だけ）
# compression_level: 1-22（高いほど圧縮率が高い）、None で圧縮無効（デフォルト: 22）
array_logger.init(root_path="./logs", compression_level=22)

# 2. 配列の種類を登録
array_logger.register(
    name="q_values",           # 配列の名前（DBファイル名になる）
    keys=["episode", "step"],  # インデックス用のキー
    shape=(10, 5),            # 配列のshape
    dtype="float32"           # データ型
)

# 3. データを記録（実験ループ内で繰り返し呼ぶ）
for episode in range(100):
    for step in range(50):
        q = np.random.rand(10, 5).astype(np.float32)
        array_logger.log(
            name="q_values",
            key_values={"episode": episode, "step": step},
            array=q
        )

# 4. 終了（全データをcommit）
array_logger.close()
```

## データの読み取り

```python
import array_logger

# リーダーを開く（圧縮ファイル .db.zst も自動的に認識・解凍）
reader = array_logger.open_reader("./logs/q_values.db")

# 全データをイテレート
for keys_dict, array in reader.iterate():
    print(f"Episode {keys_dict['episode']}, Step {keys_dict['step']}")
    print(f"Array shape: {array.shape}, dtype: {array.dtype}")
    # array を使って分析...

reader.close()
```

## API リファレンス

### `init(root_path: str)`
ロガーを初期化します。実験開始時に1回だけ呼び出します。

- `root_path`: DBファイルを保存するディレクトリ
- `compression_level`: zstd圧縮レベル（1-22、高いほど圧縮率が高いが処理時間も増加）。`None` を指定すると圧縮を無効化（デフォルト: 22）
- `root_path`: DBファイルを保存するディレクトリ

### `register(name, keys, shape, dtype, commit_threshold_rows=100, commit_threshold_seconds=60.0)`
配列の種類を登録します。各配列タイプごとに1回だけ呼び出します。

- `name`: 配列の識別名（DBファイル名: `{name}.db`）
- `keys`: インデックス用のキーのリスト（例: `["episode", "step"]`）
- `shape`: 配列のshape（タプル）
- `dtype`: データ型（文字列、例: `"float32"`）
- `commit_threshold_rows`: 自動commitする行数（デフォルト: 100）
- `commit_threshold_seconds`: 自動commitする秒数（デフォルト: 60.0）

### `log(name, array, key_values)`
データを記録します。実験ループ内で繰り返し呼び出します。

- `name`: 登録済みの配列名
- `array`: numpy ndarray（shape/dtypeは登録時と一致する必要あり）
- `key_values`: キーと値の辞書（例: `{"episode": 1, "step": 10}`）

### `close()`
ロガーを終了し、全データをcommitします。実験終了時に1回呼び出します。

### `open_reader(db_path: str) -> ArrayReader`
DBファイルからデータを読み取るリーダーを開きます。

- `db_path`: DBファイルのパス

### `ArrayReader.iterate(where_clause=None, params=None)`
データをイテレートします。

- `where_clause`: SQLのWHERE句（オプション、例: `"episode > ?"`）
- `params`: WHERE句のパラメータ（タプル）

**戻り値**: `(keys_dict, array)` のイテレータ

## よくある使い方

### 複数の配列を記録

```python
import array_logger
import numpy as np

array_logger.init("./logs")

# 異なる種類の配列を登録
array_logger.register("loss", keys=["episode"], shape=(1,), dtype="float32")
array_logger.register("weights", keys=["episode"], shape=(100, 50), dtype="float32")

for episode in range(1000):
    # lossを記録
    loss = np.array([0.5], dtype=np.float32)
    array_logger.log("loss", {"episode": episode}, loss)
    
    # weightsを記録
    weights = np.random.rand(100, 50).astype(np.float32)
    array_logger.log("weights", {"episode": episode}, weights)

array_logger.close()
```

### フィルタ付きで読み取り

```python
import array_logger

reader = array_logger.open_reader("./logs/q_values.db")

# 特定のエピソードのみ取得
for keys_dict, array in reader.iterate(where_clause="episode = ?", params=(10,)):
    print(f"Step {keys_dict['step']}: {array}")

reader.close()
```
圧縮機能

### 使用例

```python
import array_logger

# 圧縮あり（デフォルト、レベル22 = 最高圧縮率）
array_logger.init("./logs", compression_level=22)

# 圧縮あり（レベル3 = 高速・低圧縮率）
array_logger.init("./logs", compression_level=3)

# 圧縮なし
array_logger.init("./logs", compression_level=None)
```

### 圧縮の仕組み

- 書き込みの際、 `close()` 時に各 `.db` ファイルが `.db.zst` 形式に圧縮され、元の `.db` ファイルは削除されます
- 読み取り時は `.db.zst` ファイルが自動的に一時ファイルに解凍されます
- 圧縮レベルが高いほど圧縮率は向上しますが、処理時間も増加します（レベル22は非常に遅い場合があります）

### 圧縮率の目安

浮動小数点配列の場合、通常 30-70% 程度の圧縮率が期待できます（データの性質による）。

## 内部構造

各Arrayタイプごとに1つのSQLiteデータベースファイルが作成されます：

```
logs/
├── q_values.db.zst  # 圧縮あり
├── loss.db.zst      # 圧縮あり
└── weights.db       # 圧縮なし
└── weights.db
```

各DBファイル内には2つのテーブルがあります：
- `meta`: shape/dtype/keysなどのメタデータ
- `array_log`: 実際のデータ（keysのカラム + data BLOB）

詳細な設計思想は [DESIGN.md](../docs/DESIGN.md) を参照してください。

## トラブルシューティング

### `RuntimeError: Logger already initialized`
`init()` を複数回呼んでいます。実験開始時に1回だけ呼んでください。

### `ValueError: Array 'xxx' already registered`
同じ名前で `register()` を複数回呼んでいます。各配列タイプは1回だけ登録してください。

### `ValueError: Shape mismatch`
`log()` で渡した配列のshapeが `register()` 時と異なります。shapeを確認してください。

### `ValueError: Meta mismatch`
既存のDBファイルのスキーマが異なります。DBファイルを削除するか、異なる名前で登録してください。

## 要件

- Python 3.9以上
- numpy
- sqlite3（標準ライブラリ）

## ライセンス

（プロジェクトに応じて記載）
