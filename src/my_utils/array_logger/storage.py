"""
Low-level SQLite storage for a single RegisteredArray.

Responsibilities:
- SQLite connection management for one database file (e.g., "array_name.db")
- PRAGMA configuration (WAL mode for performance)
- Raw INSERT / SELECT operations
- ndarray <-> BLOB conversion

Note: Each RegisteredArray uses its own SQLiteStorage instance,
ensuring isolation between different array types.

This module MUST NOT contain:
- commit policy
- timing logic
- signal handling
"""

from typing import Any, Iterable
import sqlite3
import numpy as np

from .schema import ArraySchema


class SQLiteStorage:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self.conn: sqlite3.Connection | None = None
        self.schema: ArraySchema | None = None

    def open(self) -> None:
        """
        Open SQLite connection and apply PRAGMA settings.
        """
        if self.conn is not None:
            return  # Already open

        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")

    def close(self) -> None:
        """
        Close SQLite connection safely.
        """
        if self.conn:
            self.conn.close()
            self.conn = None

    def initialize_schema(self, schema: ArraySchema) -> None:
        """
        Create tables if they do not exist.
        """
        if self.conn is None:
            raise RuntimeError("Storage not connected")
        self.schema = schema
        self.conn.execute(schema.meta_table_sql())
        self.conn.execute(schema.log_table_sql())

    def insert_rows(
        self,
        rows: Iterable[tuple[Any, ...]],
        columns: list[str],
    ) -> None:
        """
        Insert multiple rows into the main log table.
        `columns` MUST be provided to specify column names to avoid ambiguous
        positional inserts. Each row must match the provided columns.
        """
        if self.conn is None:
            raise RuntimeError("Storage not connected")
        if self.schema is None:
            raise RuntimeError("Schema not initialized")
        rows_list = list(rows)
        if not rows_list:
            return
        col_list = ','.join(columns)
        placeholders = ','.join('?' for _ in columns)
        sql = f"INSERT INTO {self.schema.LOG_TABLE_NAME} ({col_list}) VALUES ({placeholders})"
        self.conn.executemany(sql, rows_list)

    def insert_meta(self, meta_rows: Iterable[tuple[str, str]]) -> None:
        """
        Insert rows into the meta table. Each row should be (key, value).
        """
        if self.conn is None:
            raise RuntimeError("Storage not connected")
        if self.schema is None:
            raise RuntimeError("Schema not initialized")
        rows_list = list(meta_rows)
        if not rows_list:
            return
        sql = f"INSERT INTO {self.schema.META_TABLE_NAME} (key, value) VALUES (?, ?)"
        self.conn.executemany(sql, rows_list)

    def commit(self) -> None:
        """
        Commit current transaction.
        """
        if self.conn is None:
            raise RuntimeError("Storage not connected")
        self.conn.commit()

    def checkpoint(self) -> None:
        """
        Perform WAL checkpoint.
        """
        if self.conn is None:
            raise RuntimeError("Storage not connected")
        self.conn.execute("PRAGMA wal_checkpoint(PASSIVE);")

    def select_rows(
        self,
        where_clause: str | None = None,
        params: tuple[Any, ...] | None = None,
    ) -> Iterable[tuple[Any, ...]]:
        """
        Execute SELECT on the main log table and yield rows.
        
        Note: Uses ArraySchema class constant for table name,
        so can be called even if self.schema is not fully initialized.
        """
        from .schema import ArraySchema
        
        if self.conn is None:
            raise RuntimeError("Storage not connected")
        cursor = self.conn.cursor()
        if where_clause:
            sql = f"SELECT * FROM {ArraySchema.LOG_TABLE_NAME} WHERE {where_clause}"
            cursor.execute(sql, params or ())
        else:
            sql = f"SELECT * FROM {ArraySchema.LOG_TABLE_NAME}"
            cursor.execute(sql)
        for row in cursor:
            yield row

    def select_meta(self) -> Iterable[tuple[str, str]]:
        """
        Yield rows from the meta table as (key, value).
        
        Note: Uses ArraySchema class constant for table name,
        so can be called before self.schema is initialized.
        This is essential for loading schema from metadata.
        """
        from .schema import ArraySchema
        
        if self.conn is None:
            raise RuntimeError("Storage not connected")
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT key, value FROM {ArraySchema.META_TABLE_NAME}")
        for row in cursor:
            yield row

    @staticmethod
    def ndarray_to_blob(array: np.ndarray) -> bytes:
        """
        Serialize ndarray into bytes.
        """
        return array.tobytes()

    @staticmethod
    def blob_to_ndarray(blob: bytes, dtype: str, shape: tuple[int, ...]) -> np.ndarray:
        """
        Deserialize bytes into ndarray.
        """
        return np.frombuffer(blob, dtype=dtype).reshape(shape)


class SQLiteWriter:
    """
    Write-side storage with optional compression support.
    
    This class wraps SQLiteStorage and adds compression functionality.
    On close(), it compresses the .db file to .db.zst format using zstd.
    """
    
    def __init__(self, db_path: str, compression_level: int | None) -> None:
        """
        Initialize writer with optional compression.
        
        Args:
            db_path: Path to the database file (without .zst extension)
            compression_level: Zstd compression level (1-22, higher = better compression but slower).
                              Set to None to disable compression.
        
        Raises:
            ValueError: If compression_level is not in valid range
        """
        if compression_level is not None and (compression_level < 1 or compression_level > 22):
            raise ValueError(f"compression_level must be 1-22 or None, got {compression_level}")
        
        self._storage = SQLiteStorage(db_path)
        self.compression_level = compression_level
    
    def open(self) -> None:
        """Open the database connection."""
        self._storage.open()
    
    def close(self) -> None:
        """
        Close the database connection and compress the file.
        
        Performs WAL checkpoint before compression to ensure all data is in the main file.
        Compresses the .db file to .db.zst and removes the original.
        """
        # Checkpoint WAL to merge -wal and -shm files into main db
        try:
            self._storage.checkpoint()
        except (sqlite3.OperationalError, RuntimeError):
            pass  # Checkpoint might fail if no WAL exists or storage not connected
        
        self._storage.close()
        self._compress_db()
    
    def initialize_schema(self, schema: ArraySchema) -> None:
        """Create tables if they do not exist."""
        self._storage.initialize_schema(schema)
    
    def insert_rows(self, rows: Iterable[tuple[Any, ...]], columns: list[str]) -> None:
        """Insert multiple rows into the main log table."""
        self._storage.insert_rows(rows, columns)
    
    def insert_meta(self, meta_rows: Iterable[tuple[str, str]]) -> None:
        """Insert rows into the meta table."""
        self._storage.insert_meta(meta_rows)
    
    def commit(self) -> None:
        """Commit current transaction."""
        self._storage.commit()
    
    def checkpoint(self) -> None:
        """Perform WAL checkpoint."""
        self._storage.checkpoint()
    
    def select_meta(self) -> Iterable[tuple[str, str]]:
        """Yield rows from the meta table as (key, value)."""
        return self._storage.select_meta()
    
    @property
    def schema(self) -> ArraySchema | None:
        """Get the current schema."""
        return self._storage.schema
    
    def _compress_db(self) -> None:
        """
        Compress the database file using zstd.
        
        Creates a .db.zst file and removes the original .db file.
        If compression fails, the original file is preserved.
        If compression_level is None, compression is skipped.
        """
        import os
        
        # Skip compression if disabled
        if self.compression_level is None:
            return
        
        try:
            import zstandard as zstd
        except ImportError as e:
            raise ImportError(
                "zstandard library is required for compression. "
                "Install it with: pip install zstandard"
            ) from e
        
        db_path = self._storage.db_path
        if not os.path.exists(db_path):
            return  # Nothing to compress
        
        compressed_path = f"{db_path}.zst"
        
        try:
            # Read the database file
            with open(db_path, 'rb') as f:
                data = f.read()
            
            # Compress with specified level
            compressed = zstd.compress(data, level=self.compression_level)
            
            # Write compressed file
            with open(compressed_path, 'wb') as f:
                f.write(compressed)
            
            # Remove original file only after successful compression
            os.remove(db_path)
            
            # Also remove WAL files if they exist
            for suffix in ['-wal', '-shm']:
                wal_path = f"{db_path}{suffix}"
                if os.path.exists(wal_path):
                    os.remove(wal_path)
        
        except Exception as e:
            # If compression fails, keep the original file
            if os.path.exists(compressed_path):
                try:
                    os.remove(compressed_path)
                except Exception:
                    pass
            raise RuntimeError(f"Failed to compress database: {e}") from e


class SQLiteReader:
    """
    Read-side storage with decompression support.
    
    This class wraps SQLiteStorage and adds decompression functionality.
    On open(), it automatically detects and decompresses .db.zst files to a temporary location.
    """
    
    def __init__(self, db_path: str) -> None:
        """
        Initialize reader.
        
        Args:
            db_path: Path to the database file (without .zst extension)
        """
        self.db_path = db_path
        self._temp_db_path: str | None = None
        self._storage: SQLiteStorage | None = None
    
    def open(self) -> None:
        """
        Open the database connection.
        
        If a .db.zst file exists, it will be decompressed to a temporary file.
        Otherwise, the .db file is opened directly.
        """
        actual_path = self._resolve_and_decompress()
        self._storage = SQLiteStorage(actual_path)
        self._storage.open()
    
    def close(self) -> None:
        """
        Close the database connection and clean up temporary files.
        """
        if self._storage:
            self._storage.close()
        
        # Remove temporary decompressed file if it exists
        if self._temp_db_path:
            import os
            try:
                os.remove(self._temp_db_path)
            except Exception:
                pass  # Ignore cleanup errors
            self._temp_db_path = None
    
    def select_rows(
        self,
        where_clause: str | None = None,
        params: tuple[Any, ...] | None = None,
    ) -> Iterable[tuple[Any, ...]]:
        """Execute SELECT on the main log table and yield rows."""
        if self._storage is None:
            raise RuntimeError("Reader not opened")
        return self._storage.select_rows(where_clause, params)
    
    def select_meta(self) -> Iterable[tuple[str, str]]:
        """Yield rows from the meta table as (key, value)."""
        if self._storage is None:
            raise RuntimeError("Reader not opened")
        return self._storage.select_meta()
    
    @property
    def schema(self) -> ArraySchema | None:
        """Get the current schema."""
        if self._storage is None:
            return None
        return self._storage.schema
    
    @property
    def conn(self) -> sqlite3.Connection:
        """Get the database connection (for advanced operations like to_dataframe)."""
        if self._storage is None or self._storage.conn is None:
            raise RuntimeError("Reader not opened")
        return self._storage.conn
    
    def _resolve_and_decompress(self) -> str:
        """
        Resolve the actual database path and decompress if needed.
        
        Returns:
            Path to the database file (either original .db or temporary decompressed file)
        """
        import os
        import tempfile
        
        # Check if .db file exists
        if os.path.exists(self.db_path):
            return self.db_path
        
        # Check if .db.zst file exists
        compressed_path = f"{self.db_path}.zst"
        if not os.path.exists(compressed_path):
            # Neither exists, return original path (will create new file)
            return self.db_path
        
        # Decompress .db.zst to temporary file
        try:
            import zstandard as zstd
        except ImportError as e:
            raise ImportError(
                "zstandard library is required for decompression. "
                "Install it with: pip install zstandard"
            ) from e
        
        try:
            # Read compressed file
            with open(compressed_path, 'rb') as f:
                compressed_data = f.read()
            
            # Decompress
            decompressed = zstd.decompress(compressed_data)
            
            # Create temporary file in the same directory as the original
            db_dir = os.path.dirname(self.db_path) or '.'
            db_name = os.path.basename(self.db_path)
            
            # Create temporary file with .tmp suffix
            temp_fd, temp_path = tempfile.mkstemp(
                suffix='.db',
                prefix=f'.{db_name}.tmp.',
                dir=db_dir
            )
            
            # Write decompressed data
            with os.fdopen(temp_fd, 'wb') as f:
                f.write(decompressed)
            
            self._temp_db_path = temp_path
            return temp_path
        
        except Exception as e:
            raise RuntimeError(f"Failed to decompress database: {e}") from e
