# array_logger アーキテクチャ設計書

このドキュメントは array_logger の内部構造を理解し、改良・拡張するためのガイドです。

## 目次
1. [アーキテクチャ概要](#アーキテクチャ概要)
2. [クラス構成と責務](#クラス構成と責務)
3. [データフロー](#データフロー)
4. [主要クラスの詳細設計](#主要クラスの詳細設計)
5. [設計判断の理由](#設計判断の理由)
6. [拡張ポイント](#拡張ポイント)
7. [パフォーマンス最適化](#パフォーマンス最適化)

---

## アーキテクチャ概要

### レイヤー構造

```
┌─────────────────────────────────────┐
│  Public API (array_logger/__init__)  │  ← ユーザーが使う関数
├─────────────────────────────────────┤
│  Write-side (logger.py)              │  ← データ書き込み
│  - _ArrayRegistry (private)          │
│  - ArrayRegistration                 │
├─────────────────────────────────────┤
│  Read-side (reader.py)               │  ← データ読み取り
│  - ArrayReader                       │
├─────────────────────────────────────┤
│  Schema Layer (schema.py)            │  ← スキーマ定義
│  - ArraySchema                       │
├─────────────────────────────────────┤
│  Storage Layer (storage.py)          │  ← SQLite操作
│  - SQLiteStorage                     │
└─────────────────────────────────────┘
```

### 設計原則

1. **責務の分離**: 書き込み/読み取り/スキーマ/ストレージを独立させる
2. **1配列1DB**: 配列の種類ごとに独立したDBファイル
3. **即時INSERT + 遅延commit**: 書き込み速度とデータ整合性のバランス
4. **自己記述的**: DBファイル単体でメタデータを復元可能
5. **シンプルなAPI**: 学習ループへの統合が容易

---

## クラス構成と責務

### 1. Public API (`__init__.py`)

**責務**: ユーザー向けの統一インターフェース

```python
# 公開する関数
from .logger import init, register, log, close
from .reader import ArrayReader, open_reader
```

- グローバル関数をエクスポート
- 内部クラスは隠蔽（`_ArrayRegistry` はprivate）
- `ArrayReader` のみクラスとして公開（読み取りに使用）

### 2. Write-side (`logger.py`)

#### `_ArrayRegistry` (Private Manager Class)

**責務**: 全登録配列の管理

```python
class _ArrayRegistry:
    def __init__(self, root_path: str):
        self.root_path = root_path
        self._registered_arrays: list[ArrayRegistration] = []
```

**主要メソッド**:
- `register()`: 新しい配列タイプを登録
- `log()`: データを記録（対象配列を探してINSERT）
- `close()`: 全配列をcommitしてクローズ

**設計ポイント**:
- グローバル変数 `_logger: _ArrayRegistry | None` で1つだけ存在
- 複数の `ArrayRegistration` を保持（1つの実験で複数の配列を記録）

#### `ArrayRegistration` (Per-array State)

**責務**: 1つの配列タイプの状態管理

```python
class ArrayRegistration:
    def __init__(self, name, schema, storage, ...):
        self.name = name
        self.schema = ArraySchema
        self.storage = SQLiteStorage
        
        # Commit制御
        self.commit_threshold_rows = 100
        self.commit_threshold_seconds = 60.0
        
        # 状態
        self.uncommitted_count = 0
        self.last_commit_time = time.time()
```

**主要メソッド**:
- `log_array()`: 配列を1行挿入 → `maybe_commit()` 呼び出し
- `maybe_commit()`: 閾値チェック → 必要なら `commit_pending()`
- `commit_pending()`: 実際のcommit実行

**設計ポイント**:
- 各配列タイプごとに独立した状態（uncommitted_count など）
- 最適化: `maybe_commit()` は `ArrayRegistration` オブジェクトを直接受け取る（名前検索不要）

#### グローバル関数

```python
_logger: _ArrayRegistry | None = None

def init(root_path: str):
    global _logger
    if _logger is not None:
        raise RuntimeError("Logger already initialized")
    _logger = _ArrayRegistry(root_path)

def log(name: str, array: np.ndarray, key_values: dict):
    if _logger is None:
        raise RuntimeError("Logger not initialized")
    _logger.log(name, array, key_values)
```

**設計ポイント**:
- シンプルなグローバル状態（研究コードでの使いやすさ優先）
- `_logger` の存在チェックのみ（`_initialized` フラグは不要）

### 3. Read-side (`reader.py`)

#### `ArrayReader`

**責務**: DBファイルからの読み取り

```python
class ArrayReader:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.storage = SQLiteStorage(db_path)
        self.schema: ArraySchema | None = None
        self._opened = False
    
    def open(self):
        self.storage.open()
        self._load_schema()  # 自動的にスキーマをロード
        self._opened = True
```

**主要メソッド**:
- `open()`: DB接続 + メタデータからスキーマ復元
- `iterate()`: データをイテレート（WHERE句サポート）
- `close()`: DB切断

**設計ポイント**:
- `open()` 時にスキーマを自動ロード（ユーザーは意識不要）
- Context manager サポート（`with open_reader(...) as reader:`）
- WHERE句でフィルタリング可能（特定エピソードのみ取得など）

### 4. Schema Layer (`schema.py`)

#### `ArraySchema`

**責務**: DBスキーマの定義と生成

```python
class ArraySchema:
    LOG_TABLE_NAME = "array_log"
    META_TABLE_NAME = "meta"
    
    def __init__(self, keys: list[str], shape: tuple[int, ...], dtype: str):
        self.keys = keys
        self.shape = shape
        self.dtype = dtype
```

**主要メソッド**:
- `create_table_sql()`: CREATE TABLE文の生成
- `columns()`: カラム名のリスト取得
- `to_meta_dict()` / `from_meta_dict()`: メタデータのシリアライズ

**設計ポイント**:
- テーブル名の定数を一箇所で管理（`LOG_TABLE_NAME`, `META_TABLE_NAME`）
- 動的なCREATE TABLE生成（keysの数に応じてカラムを追加）
- メタデータから完全にスキーマを復元可能

### 5. Storage Layer (`storage.py`)

#### `SQLiteStorage`

**責務**: 低レベルのSQLite操作

```python
class SQLiteStorage:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn: sqlite3.Connection | None = None
        self.schema: ArraySchema | None = None  # initialize_schema で設定
```

**主要メソッド**:
- `open()`: DB接続 + WALモード設定
- `initialize_schema()`: テーブル作成 + メタデータ挿入
- `insert_rows()`: 行の挿入（commit無し）
- `commit()`: トランザクションcommit
- `select_rows()`: 行の取得
- `insert_meta()` / `select_meta()`: メタデータ操作

**設計ポイント**:
- WALモード + `PRAGMA synchronous=NORMAL` で高速化
- `schema` を保持（テーブル名を動的に取得）
- INSERT時は即座にcommitしない（バッファリング）

---

## データフロー

### 書き込みフロー

```
ユーザーコード
    ↓ array_logger.log(name, key_values, array)
_ArrayRegistry.log()
    ↓ 該当の ArrayRegistration を検索
ArrayRegistration.log_array()
    ↓ rows = [(key1, key2, ..., blob)]
SQLiteStorage.insert_rows(rows, columns)
    ↓ INSERT INTO array_log ...（commit無し）
ArrayRegistration.maybe_commit()
    ↓ 閾値チェック
    ├→ 閾値未達 → そのまま返る
    └→ 閾値達成 → commit_pending()
        ↓
    SQLiteStorage.commit()
        ↓ COMMIT
    DB
```

**ポイント**:
1. `INSERT` は即座に実行（WALモードで他の読み取りをブロックしない）
2. `COMMIT` は閾値到達時のみ（書き込みオーバーヘッド削減）
3. 未commitデータは WAL ファイルに保持（クラッシュ時はロールバック）

### 読み取りフロー

```
ユーザーコード
    ↓ reader = array_logger.open_reader(db_path)
ArrayReader.__init__()
    ↓
ArrayReader.open()
    ↓
SQLiteStorage.open()
    ↓ 接続
SQLiteStorage.select_meta("shape", "dtype", "keys")
    ↓ SELECT * FROM meta WHERE key = ?
ArraySchema.from_meta_dict(meta)
    ↓ スキーマ復元
ArrayReader.iterate(where_clause, params)
    ↓
SQLiteStorage.select_rows(where_clause, params)
    ↓ SELECT * FROM array_log WHERE ...
    ↓ 行ごとに yield
ArrayReader._deserialize_row(row)
    ↓ blob → ndarray 変換
ユーザーコード
    ← (keys_dict, array)
```

**ポイント**:
1. `open()` 時にメタデータからスキーマを自動復元
2. WHERE句で効率的にフィルタリング可能
3. イテレータパターンでメモリ効率的

---

## 主要クラスの詳細設計

### ArraySchema のスキーマ生成

```python
def create_table_sql(self) -> str:
    # keysの数に応じて動的にカラムを生成
    key_columns = ", ".join([f"{key} INTEGER" for key in self.keys])
    
    return f"""
    CREATE TABLE IF NOT EXISTS {self.LOG_TABLE_NAME} (
        {key_columns},
        data BLOB
    )
    """
```

**拡張ポイント**: 
- keysの型を INTEGER 以外にも対応させたい場合、`keys` を `list[tuple[str, str]]` に変更
  ```python
  keys = [("episode", "INTEGER"), ("timestamp", "REAL")]
  ```

### SQLiteStorage のWAL設定

```python
def open(self):
    self.conn = sqlite3.connect(self.db_path)
    self.conn.execute("PRAGMA journal_mode=WAL")
    self.conn.execute("PRAGMA synchronous=NORMAL")
```

**設計理由**:
- `journal_mode=WAL`: 書き込み中も読み取り可能
- `synchronous=NORMAL`: 書き込み速度向上（クラッシュ時のみ一部データ損失のリスク）

**拡張ポイント**:
- より安全性を求める場合: `synchronous=FULL`
- より高速化する場合: `synchronous=OFF`（非推奨）

### ArrayRegistration のcommit戦略

```python
def maybe_commit(self):
    if (self.uncommitted_count >= self.commit_threshold_rows or
        time.time() - self.last_commit_time >= self.commit_threshold_seconds):
        self.commit_pending()

def commit_pending(self):
    self.storage.commit()
    self.uncommitted_count = 0
    self.last_commit_time = time.time()
```

**設計理由**:
- 行数ベース: 大量データを扱う場合の定期的なcommit
- 時間ベース: データが少ない場合でもデータ損失リスクを軽減

**拡張ポイント**:
- メモリ使用量ベースの閾値追加
- 適応的な閾値調整（書き込み速度に応じて動的に変更）

---

## 設計判断の理由

### なぜ1配列1DBなのか？

**判断**: 配列の種類ごとに独立したDBファイル

**理由**:
1. **スキーマの柔軟性**: 配列ごとに異なるshape/dtype/keysを持てる
2. **並列アクセス**: SQLiteのロック競合を最小化
3. **管理の容易さ**: 不要な配列のDBファイルを削除しやすい
4. **読み取りの効率**: 必要な配列のみ開ける

**トレードオフ**:
- ファイル数が増える（ディスクI/Oのオーバーヘッド）
- 複数配列の同期が必要な場合は手動で管理

### なぜ即時INSERT + 遅延commitなのか？

**判断**: `INSERT` は即実行、`COMMIT` は閾値到達時

**理由**:
1. **書き込み速度**: commitのオーバーヘッドを削減
2. **データ整合性**: WALモードで未commitデータも保護
3. **読み取りブロック回避**: WALモードで読み取りをブロックしない

**代替案との比較**:
- **毎回commit**: 安全だが遅い（数百倍の差）
- **全データバッファリング**: 高速だがクラッシュ時にデータ損失

### なぜ_ArrayRegistryをprivateにしたのか？

**判断**: `_ArrayRegistry` クラスをユーザーから隠蔽

**理由**:
1. **シンプルなAPI**: グローバル関数のみで使える
2. **誤用防止**: 複数のRegistryインスタンス作成を防ぐ
3. **柔軟性**: 内部実装を変更してもAPIに影響しない

### なぜArrayReaderは独立クラスなのか？

**判断**: 読み取り側は独立したクラスとして公開

**理由**:
1. **状態管理**: `open()`/`close()` の呼び出しが明示的
2. **複数リーダー**: 同時に複数のDBファイルを開ける
3. **Context manager**: `with` 文での安全な使用

---

## 拡張ポイント

### 1. 新しいデータ型のサポート

**現状**: numpy ndarray のみ

**拡張方法**:
```python
# ArraySchema に型変換メソッドを追加
class ArraySchema:
    def serialize(self, data):
        if isinstance(data, np.ndarray):
            return data.tobytes()
        elif isinstance(data, list):
            return pickle.dumps(data)
        # ...
    
    def deserialize(self, blob):
        # メタデータで判別
        if self.data_type == "ndarray":
            return np.frombuffer(blob, dtype=self.dtype).reshape(self.shape)
        elif self.data_type == "list":
            return pickle.loads(blob)
```

### 2. インデックスの追加

**現状**: keysにインデックス無し

**拡張方法**:
```python
# SQLiteStorage.initialize_schema() で
def initialize_schema(self, schema: ArraySchema):
    # テーブル作成
    self.conn.execute(schema.create_table_sql())
    
    # インデックス作成（新規追加）
    for key in schema.keys:
        index_sql = f"CREATE INDEX IF NOT EXISTS idx_{key} ON {schema.LOG_TABLE_NAME}({key})"
        self.conn.execute(index_sql)
```

**効果**: WHERE句による検索が高速化

### 3. 圧縮の追加

**現状**: BLOBは非圧縮

**拡張方法**:
```python
import zlib

# ArraySchema にフラグ追加
class ArraySchema:
    def __init__(self, keys, shape, dtype, compress=True):
        self.compress = compress
    
    def serialize(self, array: np.ndarray) -> bytes:
        blob = array.tobytes()
        if self.compress:
            return zlib.compress(blob)
        return blob
    
    def deserialize(self, blob: bytes) -> np.ndarray:
        if self.compress:
            blob = zlib.decompress(blob)
        return np.frombuffer(blob, dtype=self.dtype).reshape(self.shape)
```

### 4. 非同期書き込み

**現状**: 同期的なINSERT

**拡張方法**:
```python
import queue
import threading

class _ArrayRegistry:
    def __init__(self, root_path: str):
        self.root_path = root_path
        self._registered_arrays: list[ArrayRegistration] = []
        
        # 新規追加
        self.write_queue = queue.Queue()
        self.write_thread = threading.Thread(target=self._write_worker)
        self.write_thread.start()
    
    def _write_worker(self):
        while True:
            item = self.write_queue.get()
            if item is None:  # 終了シグナル
                break
            # 実際の書き込み処理
            ...
    
    def log(self, name, array, key_values):
        # キューに追加するだけ
        self.write_queue.put((name, array, key_values))
```

### 5. バリデーションの強化

**現状**: 基本的なshape/dtypeチェックのみ

**拡張方法**:
```python
class ArraySchema:
    def __init__(self, keys, shape, dtype, validators=None):
        self.validators = validators or []
    
    def validate(self, array: np.ndarray):
        # 既存のチェック
        if array.shape != self.shape:
            raise ValueError(f"Shape mismatch: {array.shape} != {self.shape}")
        
        # カスタムバリデーション
        for validator in self.validators:
            validator(array)

# 使用例
def check_positive(array):
    if np.any(array < 0):
        raise ValueError("Array must be positive")

array_logger.register(
    name="probabilities",
    keys=["step"],
    shape=(10,),
    dtype="float32",
    validators=[check_positive, lambda x: np.allclose(x.sum(), 1.0)]
)
```

---

## パフォーマンス最適化

### 現在の最適化

1. **WALモード**: 読み書き並行可能
2. **遅延commit**: 書き込み回数削減
3. **BLOB形式**: numpy配列の効率的な保存
4. **直接オブジェクト渡し**: `maybe_commit(registration)` で検索回避

### さらなる最適化案

#### 1. バッチINSERT

**現状**: 1配列ずつINSERT

**改善**:
```python
class ArrayRegistration:
    def __init__(self, ...):
        self.pending_rows = []  # バッファ
    
    def log_array(self, array, key_values):
        row = self._create_row(array, key_values)
        self.pending_rows.append(row)
        
        if len(self.pending_rows) >= BATCH_SIZE:
            self._flush_batch()
    
    def _flush_batch(self):
        self.storage.insert_rows(self.pending_rows, self.schema.columns())
        self.pending_rows.clear()
```

#### 2. プリペアドステートメント

**現状**: 毎回SQL文を構築

**改善**:
```python
class SQLiteStorage:
    def open(self):
        # ...
        self._insert_stmt = None
    
    def insert_rows(self, rows, columns):
        if self._insert_stmt is None:
            placeholders = ", ".join(["?"] * len(columns))
            sql = f"INSERT INTO {self.schema.LOG_TABLE_NAME} VALUES ({placeholders})"
            self._insert_stmt = self.conn.cursor()
            self._insert_stmt.execute(f"PREPARE stmt AS {sql}")
        
        self._insert_stmt.executemany(sql, rows)
```

#### 3. メモリマップドI/O

**現状**: 通常のファイルI/O

**改善**: 大きなDBファイルの場合、`mmap` を検討

---

## まとめ

### 主要な設計原則

1. **レイヤー分離**: API/Write/Read/Schema/Storage
2. **状態管理**: _ArrayRegistry（全体） + ArrayRegistration（個別）
3. **最適化**: 即時INSERT + 遅延commit
4. **自己記述**: メタデータからスキーマ復元

### 改良・拡張時の注意点

1. **後方互換性**: 既存のDBファイルが読めなくなるような変更は避ける
2. **テスト**: 各レイヤーごとに単体テスト作成
3. **ドキュメント**: DESIGN.md, README.md, コメントを更新
4. **パフォーマンス**: 変更後はベンチマークで確認

### 次のステップ

- [DESIGN.md](DESIGN.md): 全体仕様の確認
- [README.md](../array_logger/README.md): ユーザー向けAPI
- [examples/](../array_logger/examples/): 実用例
- テストの作成: `tests/` ディレクトリ
