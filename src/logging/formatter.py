"""
=========================================================
SleepTransitionFramework

Custom Log Formatter

Provides standardized formatting for all log messages.
=========================================================
"""

from __future__ import annotations

import logging


class ProjectFormatter(logging.Formatter):
    """
    Standard formatter used throughout the framework.
    """

    DEFAULT_FORMAT = (
        "%(asctime)s | "
        "%(levelname)-8s | "
        "%(name)s | "
        "%(message)s"
    )

    DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    # ---------------------------------------------------------

    def __init__(

        self,

        fmt: str | None = None,

        datefmt: str | None = None

    ):

        super().__init__(

            fmt=fmt or self.DEFAULT_FORMAT,

            datefmt=datefmt or self.DEFAULT_DATE_FORMAT

        )


class ConsoleFormatter(ProjectFormatter):
    """
    Formatter for console output.
    """

    def __init__(self):

        super().__init__(

            fmt="%(levelname)-8s | %(message)s"

        )


class FileFormatter(ProjectFormatter):
    """
    Formatter for log files.
    """

    def __init__(self):

        super().__init__()


class TrainingFormatter(ProjectFormatter):
    """
    Formatter used for training logs.
    """

    def __init__(self):

        super().__init__(

            fmt=(
                "%(asctime)s | "
                "Epoch %(epoch)s | "
                "%(levelname)s | "
                "%(message)s"
            )

        )


def get_console_formatter():

    return ConsoleFormatter()


def get_file_formatter():

    return FileFormatter()


def get_training_formatter():

    return TrainingFormatter()
