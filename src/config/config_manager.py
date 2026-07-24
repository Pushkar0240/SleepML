"""
=========================================================
SleepTransitionFramework

Configuration Manager

Central interface for accessing the project
configuration.

Responsibilities
----------------
- Load configuration
- Validate configuration
- Provide section access
- Reload configuration
- Save configuration
=========================================================
"""

from __future__ import annotations

from pathlib import Path

from src.config.config_loader import ConfigLoader
from src.config.config_validator import ConfigValidator


class ConfigManager:

    """
    Central configuration manager.

    Example
    -------
    >>> config = ConfigManager()

    >>> config.training["epochs"]

    >>> config.model["name"]

    >>> config.get("training", "epochs")
    """

    # -----------------------------------------------------

    def __init__(

        self,

        config_path="config/config.yaml",

        validate=True

    ):

        self.loader = ConfigLoader(config_path)

        if validate:

            ConfigValidator(

                self.loader

            ).validate()

    # -----------------------------------------------------
    # Properties
    # -----------------------------------------------------

    @property
    def project(self):

        return self.loader.get(

            "project"

        )

    @property
    def dataset(self):

        return self.loader.get(

            "dataset"

        )

    @property
    def model(self):

        return self.loader.get(

            "model"

        )

    @property
    def training(self):

        return self.loader.get(

            "training"

        )

    @property
    def callbacks(self):

        return self.loader.get(

            "callbacks"

        )

    @property
    def output(self):

        return self.loader.get(

            "output"

        )

    @property
    def logging(self):

        return self.loader.get(

            "logging"

        )

    @property
    def experiment(self):

        return self.loader.get(

            "experiment"

        )

    @property
    def random_seed(self):

        return self.loader.get(

            "random_seed"

        )

    # -----------------------------------------------------
    # General API
    # -----------------------------------------------------

    def get(

        self,

        *keys,

        default=None

    ):

        return self.loader.get(

            *keys,

            default=default

        )

    # -----------------------------------------------------

    def set(

        self,

        *keys,

        value

    ):

        self.loader.set(

            *keys,

            value=value

        )

    # -----------------------------------------------------

    def contains(

        self,

        *keys

    ):

        return self.loader.contains(

            *keys

        )

    # -----------------------------------------------------

    def save(self):

        self.loader.save()

    # -----------------------------------------------------

    def reload(self):

        self.loader.reload()

    # -----------------------------------------------------

    def summary(self):

        self.loader.summary()

    # -----------------------------------------------------

    def print(self):

        self.loader.print()

    # -----------------------------------------------------

    def to_dict(self):

        return self.loader.to_dict()

    # -----------------------------------------------------

    def __getitem__(

        self,

        key

    ):

        return self.loader.get(key)

    # -----------------------------------------------------

    def __str__(self):

        return (

            f"ConfigManager("

            f"{self.loader.path}"

            f")"

        )
