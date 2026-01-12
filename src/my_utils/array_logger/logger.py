"""
Array logging API (write-side).

This is the only module users write to.
Reading MUST NOT be implemented here.
"""

from typing import Any, Union
import time
import os
import numpy as np
import torch

from .schema import ArraySchema
from .storage import SQLiteWriter


class ArrayRegistration:
    """
    Represents a registered array type, encapsulating schema, storage, and state.

    Each RegisteredArray owns its own SQLiteWriter instance for isolation,
    ensuring one database file per array type (e.g., "loss.db", "weights.db").
    """

    def __init__(self, name: str, schema: ArraySchema, storage: SQLiteWriter, commit_threshold_rows: int, commit_threshold_seconds: float):
        self.name = name
        self.schema = schema
        self.storage = storage
        self.commit_threshold_rows = commit_threshold_rows
        self.commit_threshold_seconds = commit_threshold_seconds
        self.uncommitted_count = 0
        self.last_commit_time = 0.0


class _ArrayRegistry:
    """
    Main logger.

    Lifecycle:
    - init()
    - register()
    - log()
    - close()
    """

    def __init__(self, root_path: str, compression_level: int | None = 22) -> None:
        self.root_path = root_path
        self.compression_level = compression_level
        self._registered_arrays: list[ArrayRegistration] = []
        
        # Create root directory if it doesn't exist
        os.makedirs(root_path, exist_ok=True)

    def _find_array(self, name: str) -> ArrayRegistration:
        for reg_array in self._registered_arrays:
            if reg_array.name == name:
                return reg_array
        raise ValueError(f"Array '{name}' not registered")
    
    def _normalize_dtype(self, dtype: Union[str, np.dtype, torch.dtype]) -> str:
        """
        Convert dtype to normalized string representation.
        
        Accepts:
        - str: "float32", "int64", etc.
        - np.dtype: np.float32, np.dtype('float32'), etc.
        - torch.dtype: torch.float32, torch.int64, etc.
        
        Returns:
            Normalized dtype string (e.g., "float32", "int64")
        """
        # Handle torch.dtype
        if isinstance(dtype, torch.dtype):
            dtype = torch.zeros(1, dtype=dtype).numpy().dtype
        
        # Convert to np.dtype and normalize
        try:
            np_dtype = np.dtype(dtype)
            return np_dtype.name
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid dtype: {dtype}") from e
    
    def register(
        self,
        name: str,
        keys: list[str],
        shape: tuple[int, ...],
        dtype: Union[str, np.dtype, torch.dtype],
        commit_threshold_rows: int = 100,
        commit_threshold_seconds: float = 60.0,
    ) -> None:
        """
        Register a new array type.

        Args:
            dtype: NumPy dtype string ("float32"), np.dtype object, or torch.dtype object

        Raises:
        - RuntimeError if called after logging started
        - ValueError on invalid schema
        """
        # Check if already registered
        try:
            self._find_array(name)
            raise ValueError(f"Array '{name}' already registered")
        except ValueError:
            pass

        # Normalize dtype to string
        normalized_dtype = self._normalize_dtype(dtype)
        
        schema = ArraySchema(name, keys, shape, normalized_dtype)
        # Create dedicated storage for this RegisteredArray (one DB file per array type)
        db_path = f"{self.root_path}/{name}.db"
        
        storage = SQLiteWriter(db_path, self.compression_level)
        
        storage.open()
        storage.initialize_schema(schema)
        
        # Check meta if exists
        meta_rows = list(storage.select_meta())
        if meta_rows:
            existing_meta = {row[0]: row[1] for row in meta_rows}
            expected_meta = {
                "shape": str(shape),
                "dtype": normalized_dtype,
                "keys": str(keys),
                "array_logger_version": "1.0",  # ArrayLogger version for compatibility
            }
            if existing_meta != expected_meta:
                raise ValueError(f"Meta mismatch for array '{name}': expected {expected_meta}, got {existing_meta}")
        else:
            # Insert meta
            meta_data = [
                ("shape", str(shape)),
                ("dtype", normalized_dtype),
                ("keys", str(keys)),
                ("array_logger_version", "1.0"),  # ArrayLogger version for compatibility
            ]
            storage.insert_meta(meta_data)
            storage.commit()
        
        self._registered_arrays.append(ArrayRegistration(name, schema, storage, commit_threshold_rows, commit_threshold_seconds))

    def log(
        self,
        name: str,
        key_values: dict[str, Any],
        array: np.ndarray,
    ) -> None:
        """
        Append a array record.

        Rules:
        - name MUST be registered
        - array MUST be numpy.ndarray
        - shape/dtype MUST match registered schema
        - key_values MUST be a dict matching registered keys
        """
        reg_array = self._find_array(name)
        schema = reg_array.schema
        
        # Check array
        if not isinstance(array, np.ndarray):
            raise TypeError(f"Array must be np.ndarray, got {type(array)}")
        if array.shape != schema.shape:
            raise ValueError(f"Shape mismatch: expected {schema.shape}, got {array.shape}")
        if self._normalize_dtype(array.dtype) != schema.dtype:
            raise ValueError(f"Dtype mismatch: expected {schema.dtype}, got {self._normalize_dtype(array.dtype)}")
        
        # Check keys
        if set(key_values.keys()) != set(schema.keys):
            raise ValueError(f"Keys mismatch: expected {schema.keys}, got {list(key_values.keys())}")
        
        # Prepare row and insert immediately
        key_vals = tuple(key_values[key] for key in schema.keys)
        blob = SQLiteStorage.ndarray_to_blob(array)
        row = key_vals + (blob,)
        columns = schema.keys + ["data"]
        reg_array.storage.insert_rows([row], columns)
        
        reg_array.uncommitted_count += 1
        
        self.maybe_commit(reg_array)

    def maybe_commit(self, reg_array: ArrayRegistration) -> None:
        """
        Decide whether to commit buffered logs for this array.
        Decision is based on:
        - uncommitted count
        - elapsed time since last commit
        """
        now = time.time()
        if (reg_array.uncommitted_count >= reg_array.commit_threshold_rows or
            now - reg_array.last_commit_time >= reg_array.commit_threshold_seconds):
            self.commit_pending(reg_array)

    def commit_pending(self, reg_array: ArrayRegistration) -> None:
        """
        Force commit pending logs for a specific array and reset counters.
        """
        reg_array.storage.commit()
        reg_array.uncommitted_count = 0
        reg_array.last_commit_time = time.time()

    def close(self) -> None:
        """
        Commit all pending logs and close all storages.
        Safe to call multiple times.
        """
        for reg_array in self._registered_arrays:
            self.commit_pending(reg_array)
            reg_array.storage.close()
        self._registered_arrays.clear()
    
    def list_registered(self) -> list[dict[str, Any]]:
        """
        Get list of registered arrays.
        
        Returns:
            List of dictionaries with array information. Currently only "name" key is included.
            Example: [{"name": "q_values"}, {"name": "loss"}]
        """
        return [{"name": reg_array.name} for reg_array in self._registered_arrays]


# Global logger instance
_logger: _ArrayRegistry | None = None


def init(run_id: str, compression_level: int | None = 22) -> None:
    """
    Initialize the global logger.
    
    Args:
        run_id: Unique identifier for the run. Typically a WandB run ID (e.g., "2ip2oxwn"),
                but can be any unique string identifier.
        compression_level: Zstd compression level (1-22, higher = better compression but slower).
                          Set to None to disable compression. Default: 22.
    
    Example:
        >>> import array_logger
        >>> array_logger.init(run_id="2ip2oxwn")
    """
    from . import utils
    
    global _logger
    if _logger is not None:
        raise RuntimeError("Logger already initialized")
    
    root_path = utils.run_path(run_id)
    _logger = _ArrayRegistry(root_path, compression_level)


def register(
    name: str,
    keys: list[str],
    shape: tuple[int, ...],
    dtype: str,
    commit_threshold_rows: int = 100,
    commit_threshold_seconds: float = 60.0,
) -> None:
    """
    Register a array type.
    """
    if _logger is None:
        raise RuntimeError("Logger not initialized")
    _logger.register(name, keys, shape, dtype, commit_threshold_rows, commit_threshold_seconds)


def log(name: str, key_values: dict[str, Any], array: np.ndarray) -> None:
    """
    Log a array.
    """
    if _logger is None:
        raise RuntimeError("Logger not initialized")
    _logger.log(name, key_values, array)


def close() -> None:
    """
    Close the global logger.
    """
    global _logger
    if _logger is not None:
        _logger.close()
        _logger = None


def list_registered() -> list[dict[str, Any]]:
    """
    Get list of registered arrays.
    
    Returns:
        List of dictionaries with array information. Currently only "name" key is included.
        Example: [{"name": "q_values"}, {"name": "loss"}]
    
    Raises:
        RuntimeError: If logger is not initialized
    
    Example:
        >>> import array_logger
        >>> array_logger.init(run_id="test")
        >>> array_logger.register("q_values", keys=["episode"], shape=(10,), dtype="float32")
        >>> arrays = array_logger.list_registered()
        >>> print(arrays)
        [{"name": "q_values"}]
    """
    if _logger is None:
        raise RuntimeError("Logger not initialized")
    return _logger.list_registered()
