"""SleepTransitionCNN
==================
Logging Utility

Features
--------
1. Console logging
2. File logging
3. Colored log levels (future extension)
4. Timestamped log files
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from src.app_config import config


class Logger:
    """Central logging utility for the entire project.

    Example
    -------
    >>> logger = Logger().get_logger("Dataset")
    >>> logger.info("Loading EDF...")
    """

    def __init__(self) -> None:
        self.log_directory: Path = config.logs
        self.log_directory.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_directory / f"run_{timestamp}.log"

    def get_logger(self, name: str) -> logging.Logger:
        """Return a configured logger for the given component name."""

        logger = logging.getLogger(name)

        # Prevent duplicate handlers.
        if logger.handlers:
            return logger

        logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)-18s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Console output.
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        # File output.
        file_handler = logging.FileHandler(self.log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        logger.propagate = False
        return logger


# Singleton logger factory.
logger_factory = Logger()
