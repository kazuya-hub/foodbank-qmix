"""
Database schema definition.

This file defines the authoritative SQLite schema.
Do NOT put runtime logic here.
"""


class ArraySchema:
    """
    Schema for a single array type (one database file).
    """

    LOG_TABLE_NAME = "array_log"
    META_TABLE_NAME = "meta"

    def __init__(
        self,
        name: str,
        keys: list[str],
        shape: tuple[int, ...],
        dtype: str,
    ) -> None:
        self.name = name
        self.keys = keys
        self.shape = shape
        self.dtype = dtype

    def meta_table_sql(self) -> str:
        """
        Return CREATE TABLE SQL for meta information.
        """
        return f"""
CREATE TABLE IF NOT EXISTS {self.META_TABLE_NAME} (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

    def log_table_sql(self) -> str:
        """
        Return CREATE TABLE SQL for array log rows.

        NOTE:
        - shape and dtype MUST NOT be included here, as they are in meta table.
        """
        columns = []
        for key in self.keys:
            columns.append(f"{key} INTEGER NOT NULL")
        columns.append("data BLOB NOT NULL")
        column_defs = ",\n    ".join(columns)
        return f"""
CREATE TABLE IF NOT EXISTS {self.LOG_TABLE_NAME} (
    {column_defs}
);
"""
