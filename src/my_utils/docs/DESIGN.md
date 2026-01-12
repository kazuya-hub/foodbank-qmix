# array_logger 設計ドキュメント

## 目的

本モジュールは、強化学習などの実験中に発生する **Array データ（numpy ndarray）を効率よく蓄積・永続化**するためのロガーである。

- 実験中は **高速・低オーバーヘッド**
- 実験終了時に **まとめて DB に commit**
- Array の種類ごとに **独立した管理**
- DB 単体で **自己記述的（shape / dtype が分かる）**

---

## ユーザーAPI

```python
import array_logger

array_logger.init(root_path="logs/") # ロガーを初期化する。実験中に一度だけ

array_logger.register( # Arrayの種類を登録する。実験中、Array種類ごとに1度だけ
    name="q_values",
    keys=["episode", "step"],
    shape=(320, 100),
    dtype="float32",
)

array_logger.log( # ログを追記する。実験中に繰り返し呼び出される
    name="q_values",
    array=q,
    key_values={"episode": ep, "step": t}
)

array_logger.close() # ロガーを終了し、データを確定させる。実験中に一度だけ
```

## ディレクトリ構造

```text
docs/
├─ DESIGN.md (このファイル)
array_logger/
├─ __init__.py
├─ logger.py
├─ reader.py
├─ storage.py
└─ schema.py
```

### 各ファイルの責務

#### `__init__.py`

- ユーザーに公開する API のみを export
- 内部構造（storage / schema）は隠蔽する

#### `logger.py`（書き込み側・上位API）

- ユーザーが直接呼び出す記録用 API
- `init / register / log / close`
- Array 種類ごとの runtime 状態管理  
    （未 commit 件数、最終 commit 時刻など）
- commit 判定（件数 + 時間）
- 例外・Ctrl+C・atexit 時の安全な flush

**禁止事項**

- DB スキーマ定義
- SQL の直接記述
- 読み取り処理

#### `reader.py`（読み取り側・分析API）

- DB から Array を読み取る専用 API
- フィルタ条件付きイテレーション（例: keys に応じた WHERE 句でフィルタ）
- BLOB → ndarray 復元
- 分析者が使う入口

**禁止事項**

- 書き込み（INSERT / COMMIT）
- commit / checkpoint

#### `storage.py`（低レベル永続化）

- SQLite 接続管理
- PRAGMA 設定（WAL 等）
- INSERT / SELECT / COMMIT / CHECKPOINT
- ndarray ↔ BLOB 変換

**禁止事項**

- commit 判定ロジック
- Array の意味的解釈
- timing / signal 処理

#### `schema.py`（DB 仕様の唯一の定義）

- SQLite テーブル構造の定義
- meta テーブル（shape / dtype / keys）
- log テーブル（keys + data のみ）
- 「shape / dtype は行ごとに保存しない」という設計制約を明示

**禁止事項**

- DB 接続
- 実行時ロジック
- INSERT / SELECT

### 設計上の重要原則（短縮版）

- **1 DB = 1 Array 種類**
- shape / dtype は meta にのみ保存
- 書き込み（logger）と読み取り（reader）は分離
- DB 仕様は schema.py に集中させる

---

## Array 単位の管理方針（重要）

### commit 判定は「Array 種類ごと」

❌ 実験全体で一括管理しない  
✅ Array 種類ごとに以下を保持する

- 未 commit 行数
- 最終 commit 時刻
- commit 閾値（行数 or 時刻） - register() のデフォルト引数で指定、デフォルトは行数 100、秒数 60.0

---

## register の仕様

### シグネチャ（例）

```python
register(
    name: str,
    keys: list[str],
    shape: tuple[int, ...],
    dtype: str,
    commit_threshold_rows: int = 100,
    commit_threshold_seconds: float = 60.0,
)
```

### register 時に決定する情報（不変）

- Array 名（= DB ファイル名）
- shape（固定）
- dtype（固定、文字列）
- keys（例: ["episode", "step"]） - これに基づき array_log テーブルのカラムを動的に決定
- commit 閾値（行数: commit_threshold_rows, 秒数: commit_threshold_seconds） - array_logger が管理

👉 **これらは log 時には変更不可**

---

## dtype を文字列で受け取る理由と扱い

### なぜ文字列か

- JSON / SQLite / メタデータとして保存可能
- NumPy / PyTorch に依存しない表現

### 内部実装ルール

- register 時に `np.dtype(dtype)` を試みる
- 失敗した場合は **即例外**
- 正規化された dtype 文字列を内部保存する

```python
np_dtype = np.dtype(dtype)
normalized_dtype = str(np_dtype)
```

---

## log() の引数設計

### log が受け取る型

✅ **NumPy ndarray を受け取る**  
❌ PyTorch Tensor は直接受け取らない

理由：

- SQLite / NumPy との相性が良い
- フレームワーク依存を排除
- 呼び出し側で `.detach().cpu().numpy()` すれば良い

### log 時のチェック

- 型が `np.ndarray` であること
- shape が register 時と一致
- dtype が register 時と一致

---

## DB 設計（Array 種類ごとに1 DB）

### テーブル構成

#### meta テーブル（1行だけ）

```sql
CREATE TABLE meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
```

保存内容：

|key|value|
|---|---|
|shape|JSON 文字列|
|dtype|正規化された dtype|
|keys|JSON 配列|
|array_logger_version|array_logger の version|

---

#### array_log テーブル（本体）

```sql
CREATE TABLE array_log (
    -- keys に応じて動的にカラムを決定（例: episode INTEGER NOT NULL, step INTEGER NOT NULL）
    data BLOB NOT NULL
);
```

※ keys は register 時に指定されたものを使用し、テーブルカラムを動的に生成

---

### 重要な禁止事項

❌ shape / dtype を各行に保存しない  
❌ meta を Python 側だけに持たない

👉 **DB 単体で復元可能であることが必須**

---

## データの保存形式

### data カラム

- `ndarray.tobytes()` をそのまま保存
- エンディアンは NumPy に委ねる
- 圧縮は行わない（将来拡張）

### 復元方法（参考）

```python
np.frombuffer(blob, dtype).reshape(shape)
```

---

## commit / flush の仕様

### 内部バッファ

- 各 Array ごとに即座に INSERT を実行（WAL モード活用）
- log() は **DB に即書き込むが、commit は遅延**

### commit 条件（Array ごと）

- 未 commit 行数 >= commit_threshold_rows
- 最終 commit から commit_threshold_seconds 秒経過

### close()

- 全 Array の未 commit 分を **強制 commit**
- 全ストレージ接続をクローズ

---

## エラー時の挙動

- register 時に meta が存在する場合
    - 内容が完全一致 → OK
    - 不一致 → 例外（誤用防止）
- log 時の shape / dtype 不一致 → 例外

---

## 実装上の注意（Copilot 向け）

- SQLite 接続は Array ごとに管理して良い
- commit 頻度を下げることを最優先
- コードの可読性 > マイクロ最適化
- 例外メッセージは **具体的に**

---

## 将来拡張（今は実装しない）

- 圧縮（zstd 等）
- Parquet への変換
- 非同期 commit

---

## 実装完了の定義

- register → log → close → DB 単体で復元可能
- Array 種類を増やしても相互に干渉しない
- 実験途中でプロセスが落ちても DB が壊れない

---

## 実装タスクリスト

### 準備フェーズ

- [x] 必要なライブラリの確認とインストール（NumPy, SQLite3 - Python 標準）
- [x] プロジェクト構造の確認（array_logger/ ディレクトリ作成済み）

### コア実装フェーズ

- [x] schema.py の実装
    - [x] meta テーブルの CREATE 文定義
    - [x] array_log テーブルの動的 CREATE 文生成関数（keys に応じて）
    - [x] meta データの INSERT/SELECT 文定義
- [x] storage.py の実装
    - [x] SQLite 接続管理クラス（Array ごとに接続）
    - [x] PRAGMA 設定（WAL モード等）
    - [x] ndarray ↔ BLOB 変換関数（tobytes/frombuffer）
    - [x] INSERT/SELECT/COMMIT/CHECKPOINT メソッド
- [x] logger.py の実装
    - [x] _ArrayRegistry クラス設計（プライベートクラス + グローバルインスタンス）
    - [x] init() メソッド（root_path 設定）
    - [x] register() メソッド（meta チェック、DB 初期化、commit 閾値保存）
    - [x] log() メソッド（ndarray チェック、バッファ蓄積、commit 判定）
    - [x] close() メソッド（全 Array の commit と storage close）
    - [x] 内部状態管理（Array ごとの ArrayRegistration リスト）
    - [x] commit 判定ロジック（行数 + 時間）
- [x] reader.py の実装
    - [x] DB から meta 読み取り関数
    - [x] フィルタ付きイテレーション関数（keys に応じた WHERE 句生成）
    - [x] BLOB → ndarray 復元関数
- [x] __init__.py の実装
    - [x] グローバル関数（init, register, log, close）と ArrayReader の export

### 指摘事項対応フェーズ

- [x] クラス設計の見直し
    - [x] ArrayRegistration クラスを新たに追加（各Array種類のschema, storage, stateをまとめる）
    - [x] _ArrayRegistry の dict を ArrayRegistration の list に変更
    - [x] _initialized フラグを削除し、シンプル化
- [x] 小規模な仕様の誤解修正
    - [x] log() の仕様変更: buffer append ではなく即 insert（WALモード利用）
    - [x] maybe_commit() の仕様変更: insert 不要、commit のみ
    - [x] close() に storage.close() の呼び出しを追加
    - [x] meta の所在統一: 各Array DBファイル内の meta テーブルに確定
- [x] 雑多な修正提案対応
    - [x] storage.py に schema 保持機能を追加し、テーブル名を ArraySchema から取得
    - [x] storage.py の全メソッドに schema 存在チェックを追加
    - [x] logger.py register() の version キーを array_logger_version に変更
    - [x] log() の引数名 tensor を array、keys を key_values に変更
    - [x] _ArrayRegistry をプライベートクラス化
    - [x] maybe_commit/commit_pending を最適化（name ではなく ArrayRegistration を渡す）

### テストフェーズ

- [ ] 基本的なテスト環境のセットアップ（unittest または pytest）
- [ ] 単体テスト作成
    - [ ] register() の正常系（新規 DB 作成、meta 保存）
    - [ ] register() の異常系（meta 不一致例外）
    - [ ] log() の正常系（バッファ蓄積、commit 判定）
    - [ ] log() の異常系（shape/dtype 不一致例外）
    - [ ] close() のテスト（全 flush）
    - [ ] reader のテスト（DB 復元、フィルタイテレーション）
- [ ] 統合テスト
    - [ ] register → log → close の一連フロー
    - [ ] 複数 Array の同時管理
    - [ ] commit 閾値の動作確認（行数/時間）
- [ ] エッジケーステスト
    - [ ] プロセス強制終了時の DB 整合性
    - [ ] 大規模 Array の保存/読み取り
    - [ ] keys の異なるパターン（["episode"], ["time", "batch"] 等）

### バリデーションフェーズ

- [ ] コード品質チェック
    - [ ] リンター実行（flake8, black 等）
    - [ ] 型ヒントの確認（mypy 等）
- [ ] パフォーマンス検証
    - [ ] commit 頻度の測定
    - [ ] 大量データ時のメモリ使用量
- [ ] ドキュメント更新
    - [ ] README.md 作成（使用例、API リファレンス）
    - [ ] DESIGN.md の実装完了定義確認

### 最終確認フェーズ

- [ ] 実装完了定義の検証
    - [ ] DB 単体復元テスト
    - [ ] Tensor 種類の独立性確認
    - [ ] プロセス落ち時の DB 破損なし
- [ ] リリース準備
    - [ ] バージョン管理（**version** 定義）
    - [ ] 依存関係の明記（requirements.txt 等）
