"""Structured logging setup for EventPulse."""

import logging
import os
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "eventpulse",
    log_file: Optional[str] = "logs/eventpulse.log",
    level: str = "INFO",
    log_format: Optional[str] = None,
    date_format: Optional[str] = None,
) -> logging.Logger:
    """Configure and return a structured logger.

    Args:
        name: Logger name.
        log_file: Path to destination log file. If None, only logs to stdout.
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR').
        log_format: Custom log format string.
        date_format: Custom date format string.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers if setup is called multiple times
    if logger.handlers:
        return logger

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    default_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    default_date_format = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(
        fmt=log_format or default_format,
        datefmt=date_format or default_date_format,
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if requested)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger
