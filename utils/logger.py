# utils/logger.py

import os
import logging
from logging.handlers import RotatingFileHandler

# Configuration from environment or defaults
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_PATH = os.getenv("LOG_PATH", "logs/app.log")
MAX_BYTES = 10 * 1024 * 1024  # 10 MB per log file
BACKUP_COUNT = 5              # keep last 5 log files

# Ensure log directory exists
log_dir = os.path.dirname(LOG_PATH)
if log_dir and not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)

# Formatter
_formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Console handler
_console_handler = logging.StreamHandler()
_console_handler.setLevel(LOG_LEVEL)
_console_handler.setFormatter(_formatter)

# Rotating file handler
_file_handler = RotatingFileHandler(
    LOG_PATH,
    maxBytes=MAX_BYTES,
    backupCount=BACKUP_COUNT,
    encoding="utf-8"
)
_file_handler.setLevel(LOG_LEVEL)
_file_handler.setFormatter(_formatter)

# Root logger configuration
_root_logger = logging.getLogger()
_root_logger.setLevel(LOG_LEVEL)
_root_logger.addHandler(_console_handler)
_root_logger.addHandler(_file_handler)


def get_logger(name: str = None) -> logging.Logger:
    """
    Retrieve a logger instance with the given name.
    If name is None, returns the root logger.
    """
    return logging.getLogger(name)
