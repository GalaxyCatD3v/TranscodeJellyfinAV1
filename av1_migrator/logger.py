"""
Logging module for Galaxy AV1 Migrator.
Provides rotating file logs and console/rich logging handlers.
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional
from rich.logging import RichHandler


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
        rich_handler = RichHandler(
            level=logging.DEBUG if verbose else logging.INFO,
            show_time=True,
            show_path=False,
            markup=True,
        )
        logger.addHandler(rich_handler)

    _logger = logger
    return logger


def get_logger() -> logging.Logger:
    global _logger
    if _logger is None:
        _logger = setup_logger()
    return _logger
