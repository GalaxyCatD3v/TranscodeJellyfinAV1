"""
Logging module for Galaxy AV1 Migrator.
Provides rotating file logs and tqdm-safe console logging handlers.
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional
from tqdm import tqdm


class TqdmLoggingHandler(logging.Handler):
    """
    Logging handler that routes console output through tqdm.write
    to ensure active multi-line progress bars are never corrupted or displaced.
    """
    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


_logger: Optional[logging.Logger] = None


def setup_logger(
    log_dir: str | Path = "logs",
    filename: str = "av1-migrator.log",
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    verbose: bool = False,
    enable_console: bool = False,
) -> logging.Logger:
    global _logger
    
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)
    log_file = log_dir_path / filename

    logger = logging.getLogger("av1_migrator")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()

    # Rotating file handler
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)-8s] [%(name)s:%(module)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if enable_console:
        tqdm_handler = TqdmLoggingHandler(level=logging.DEBUG if verbose else logging.INFO)
        tqdm_formatter = logging.Formatter("[%(asctime)s] [%(levelname)-8s] %(message)s", datefmt="%H:%M:%S")
        tqdm_handler.setFormatter(tqdm_formatter)
        logger.addHandler(tqdm_handler)

    _logger = logger
    return logger


def get_logger() -> logging.Logger:
    global _logger
    if _logger is None:
        _logger = setup_logger()
    return _logger
