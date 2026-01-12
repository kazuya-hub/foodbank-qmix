"""
Utility functions for array_logger.

This module contains file system operations and other utility functions
that don't fit into logger, reader, storage, or schema modules.
"""

import os
import glob
from typing import Any


def run_path(run_id: str) -> str:
    """
    Construct the full path to a run's array_logger directory.
    
    Args:
        run_id: Unique identifier for the run. Typically a WandB run ID (e.g., "2ip2oxwn"),
                but can be any unique string identifier.
    
    Returns:
        Full path to the run directory (e.g., "results/array_logs/2ip2oxwn")
    
    Example:
        >>> import array_logger
        >>> path = array_logger.utils.run_path("2ip2oxwn")
        >>> print(path)
        results/array_logs/2ip2oxwn
    """
    return os.path.join("results", "array_logs", run_id)


def list_logged_arrays(run_id: str) -> list[dict[str, Any]]:
    """
    List all logged arrays for the specified run ID.
    
    Scans for .db and .db.zst files and returns array names without extensions.
    
    Args:
        run_id: Unique identifier for the run. Typically a WandB run ID (e.g., "2ip2oxwn"),
                but can be any unique string identifier.
    
    Returns:
        List of dictionaries with array information. Currently only "name" key is included.
        Example: [{"name": "loss"}, {"name": "q_values"}, {"name": "weights"}]
        
        Future extensions may add keys like "size_mb", "last_modified", etc.
    
    Raises:
        FileNotFoundError: If the run directory does not exist
        NotADirectoryError: If the path exists but is not a directory
        PermissionError: If there are insufficient permissions to access the directory
    
    Example:
        >>> import array_logger
        >>> arrays = array_logger.utils.list_logged_arrays("2ip2oxwn")
        >>> print(arrays)
        [{"name": "loss"}, {"name": "q_values"}]
        >>> 
        >>> # Check if specific array exists
        >>> array_names = [arr["name"] for arr in arrays]
        >>> if "q_values" in array_names:
        ...     print("q_values log found")
    """
    # Get run directory path
    path = run_path(run_id)
    
    # Check if path exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"Run directory not found: {path}")
    
    # Check if path is a directory
    if not os.path.isdir(path):
        raise NotADirectoryError(f"Path is not a directory: {path}")
    
    # Collect array names from .db and .db.zst files
    array_names = set()
    
    # Find .db files
    try:
        db_files = glob.glob(os.path.join(path, "*.db"))
        for file_path in db_files:
            # Remove .db extension (last 3 characters)
            name = os.path.basename(file_path)[:-3]
            array_names.add(name)
        
        # Find .db.zst files
        zst_files = glob.glob(os.path.join(path, "*.db.zst"))
        for file_path in zst_files:
            # Remove .db.zst extension (last 7 characters)
            name = os.path.basename(file_path)[:-7]
            array_names.add(name)
    except PermissionError as e:
        raise PermissionError(f"Permission denied accessing directory: {path}") from e
    
    # Return as list of dictionaries (no sorting)
    return [{"name": name} for name in array_names]
