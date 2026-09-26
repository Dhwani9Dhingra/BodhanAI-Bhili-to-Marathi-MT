"""Central logging configuration."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

LOGGER_NAME = "bodhan_bhili"


def setup_logging(log_file: str | Path, level: int = logging.INFO) -> logging.Logger:
    """Configure and return the project logger."""
    log_file = Path(log_file)

    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(LOGGER_NAME)

    logger.setLevel(level)

    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)

        handler.close()

    formatter = logging.Formatter(
        fmt=("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"), datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler(sys.stdout)

    console_handler.setLevel(level)

    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(filename=log_file, mode="a", encoding="utf-8")

    file_handler.setLevel(level)

    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    logger.addHandler(file_handler)

    return logger


def get_logger() -> logging.Logger:
    """Get the shared project logger."""
    return logging.getLogger(LOGGER_NAME)
