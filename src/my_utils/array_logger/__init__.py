from .logger import init, register, log, close, list_registered
from .reader import ArrayReader
from . import utils

def open_reader(db_path: str) -> ArrayReader:
    """
    Open a reader for an array database.
    """
    reader = ArrayReader(db_path)
    reader.open()
    return reader