"""
=========================================================
SleepTransitionFramework

Logger

Central logging system used across the project.

Features
--------
- Console logging
- File logging
- Timestamped messages
- Configurable log level
=========================================================
"""

from __future__ import annotations

import logging
from pathlib import Path


class ProjectLogger:

    """
    Central logger for the framework.

    Example
    -------
    logger = ProjectLogger().get_logger()

    logger.info("Training started.")
    """

    _instance = None

    # -----------------------------------------------------

    def __new__(cls):

        if cls._instance is None:

            cls._instance = super().__new__(cls)

        return cls._instance

    # -----------------------------------------------------

    def __init__(

        self,

        name="SleepTransitionFramework",

        log_dir="outputs/logs",

        level=logging.INFO

    ):

        if hasattr(self, "_initialized"):

            return

        self._initialized = True

        self.logger = logging.getLogger(name)

        self.logger.setLevel(level)

        self.logger.propagate = False

        Path(log_dir).mkdir(

            parents=True,

            exist_ok=True

        )

        log_file = Path(log_dir) / "project.log"

        from src.logging.formatter import (
            get_console_formatter,
            get_file_formatter
        )

        console_handler = logging.StreamHandler()

        console_handler.setFormatter(
            get_console_formatter()
        )

        file_handler = logging.FileHandler(

            log_file,

            mode="a",

            encoding="utf-8"

        )

        file_handler.setFormatter(
            get_file_formatter()
        )

        if not self.logger.handlers:

            self.logger.addHandler(console_handler)

            self.logger.addHandler(file_handler)

    # -----------------------------------------------------

    def get_logger(self):

        return self.logger
