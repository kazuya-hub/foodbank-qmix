"""
Array log reader (read-side API).

This module MUST NOT perform any writes.
"""

from typing import Any, Iterable
import numpy as np

from .schema import ArraySchema
from .storage import SQLiteStorage, SQLiteReader


class ArrayReader:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self.storage: SQLiteReader | None = None
        self.schema: ArraySchema | None = None

    def open(self) -> None:
        """
        Open database for reading and load schema.
        """
        self.storage = SQLiteReader(self.db_path)
        self.storage.open()
        self._load_schema()

    def close(self) -> None:
        """
        Close reader.
        """
        if self.storage:
            self.storage.close()

    def _load_schema(self) -> None:
        """
        Load array schema from meta table (internal use only).
        """
        if self.storage is None:
            raise RuntimeError("Reader not opened")
        
        meta_rows = list(self.storage.select_meta())
        meta_dict = {row[0]: row[1] for row in meta_rows}
        
        import ast
        shape = ast.literal_eval(meta_dict["shape"])
        dtype = meta_dict["dtype"]
        keys = ast.literal_eval(meta_dict["keys"])
        name = "unknown"  # Not stored, but needed for schema
        
        self.schema = ArraySchema(name, keys, shape, dtype)

    def iterate(
        self,
        where_clause: str | None = None,
        params: tuple[Any, ...] | None = None,
    ) -> Iterable[tuple[dict[str, Any], np.ndarray]]:
        """
        Iterate over logged arrays.

        Yields:
        - keys dict
        - ndarray
        """
        if self.schema is None:
            raise RuntimeError("Reader not opened")
        if self.storage is None:
            raise RuntimeError("Reader not opened")
        
        for row in self.storage.select_rows(where_clause, params):
            key_vals = row[:-1]
            blob = row[-1]
            keys_dict = dict(zip(self.schema.keys, key_vals))
            array = SQLiteStorage.blob_to_ndarray(blob, self.schema.dtype, self.schema.shape)
            yield keys_dict, array

    def to_dataframe(self):
        """
        全データを pandas DataFrame に変換
        
        Returns:
            DataFrame with columns: [key1, key2, ..., 'array']
            - keys列: int/float など
            - 'array'列: numpy.ndarray オブジェクト（元の shape, dtype を保持）
            
        Example:
            >>> df = reader.to_dataframe()
            >>> df.head()
               episode  t_env  t_ep                    array
            0        1    100     5  [[1, 0, 1], [0, 1, 1]]
            1        1    101     6  [[1, 1, 0], [0, 0, 1]]
            
            >>> # エピソードごとに集計
            >>> episode_avg = df.groupby('episode')['array'].apply(
            ...     lambda x: np.mean(np.stack(x), axis=0)
            ... )
        """
        import pandas as pd
        
        if self.schema is None:
            raise RuntimeError("Reader not opened")
        if self.storage is None:
            raise RuntimeError("Reader not opened")
        
        # 1. Keys部分を高速読み込み
        key_cols = ', '.join(self.schema.keys)
        query = f"SELECT {key_cols} FROM {ArraySchema.LOG_TABLE_NAME}"
        df = pd.read_sql_query(query, self.storage.conn)
        
        # 2. BLOB データを一括取得・変換
        blob_query = f"SELECT data FROM {ArraySchema.LOG_TABLE_NAME}"
        cursor = self.storage.conn.execute(blob_query)
        rows = cursor.fetchall()
        
        # 全BLOBを結合して一度に変換（高速）
        buf = b"".join(row[0] for row in rows)
        arrays = np.frombuffer(buf, dtype=self.schema.dtype).reshape(
            len(df), *self.schema.shape
        )
        
        # 3. ndarray列を追加
        df['array'] = list(arrays)
        
        return df
