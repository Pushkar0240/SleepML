"""
=========================================================
SleepTransitionFramework

Configuration Loader

Loads, validates and manages YAML configuration files.
=========================================================
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


class ConfigLoader:
    """
    Loads and manages project configuration.

    Example
    -------
    >>> config = ConfigLoader()
    >>> epochs = config.get("training", "epochs")
    >>> lr = config.get("training", "learning_rate")
    """

    # ---------------------------------------------------------

    def __init__(

        self,

        config_path: str | Path = "config/config.yaml"

    ):

        self.config_path = Path(config_path)

        self._config = {}

        self.load()

    # ---------------------------------------------------------

    @property
    def path(self):

        return self.config_path

    # ---------------------------------------------------------

    def exists(self):

        return self.config_path.exists()

    # ---------------------------------------------------------

    def load(self):

        """
        Load YAML configuration.
        """

        if not self.exists():

            raise FileNotFoundError(

                f"Configuration file not found:\n"

                f"{self.config_path}"

            )

        with open(

            self.config_path,

            "r",

            encoding="utf-8"

        ) as file:

            self._config = yaml.safe_load(file)

        if self._config is None:

            self._config = {}

        return self._config

    # ---------------------------------------------------------

    def reload(self):

        """
        Reload configuration from disk.
        """

        return self.load()

    # ---------------------------------------------------------

    def save(self):

        """
        Save configuration.
        """

        with open(

            self.config_path,

            "w",

            encoding="utf-8"

        ) as file:

            yaml.safe_dump(

                self._config,

                file,

                sort_keys=False,

                allow_unicode=True

            )

    # ---------------------------------------------------------

    def get(

        self,

        *keys,

        default=None

    ):

        """
        Nested configuration lookup.

        Example

        config.get(
            "training",
            "epochs"
        )
        """

        value = self._config

        for key in keys:

            if isinstance(value, dict):

                value = value.get(

                    key,

                    default

                )

            else:

                return default

        return value

    # ---------------------------------------------------------

    def set(

        self,

        *keys,

        value

    ):

        """
        Update nested value.

        Example

        config.set(
            "training",
            "epochs",
            value=100
        )
        """

        node = self._config

        for key in keys[:-1]:

            if key not in node:

                node[key] = {}

            node = node[key]

        node[keys[-1]] = value

    # ---------------------------------------------------------

    def contains(

        self,

        *keys

    ):

        """
        Check if key exists.
        """

        value = self._config

        for key in keys:

            if not isinstance(value, dict):

                return False

            if key not in value:

                return False

            value = value[key]

        return True

    # ---------------------------------------------------------

    def remove(

        self,

        *keys

    ):

        """
        Remove configuration key.
        """

        node = self._config

        for key in keys[:-1]:

            node = node[key]

        node.pop(

            keys[-1],

            None

        )

    # ---------------------------------------------------------

    def to_dict(self):

        """
        Return deep copy.
        """

        return deepcopy(

            self._config

        )

    # ---------------------------------------------------------

    def keys(self):

        return self._config.keys()

    # ---------------------------------------------------------

    def sections(self):

        """
        List top-level sections.
        """

        return list(

            self._config.keys()

        )

    # ---------------------------------------------------------

    def print(self):

        """
        Pretty print configuration.
        """

        print()

        print("=" * 70)

        print("PROJECT CONFIGURATION")

        print("=" * 70)

        print()

        print(

            yaml.dump(

                self._config,

                sort_keys=False,

                default_flow_style=False

            )

        )

    # ---------------------------------------------------------

    def summary(self):

        print()

        print("=" * 70)

        print("Configuration Summary")

        print("=" * 70)

        print()

        print(

            "Project :",

            self.get(

                "project",

                "name"

            )

        )

        print(

            "Version :",

            self.get(

                "project",

                "version"

            )

        )

        print()

        print(

            "Model :",

            self.get(

                "model",

                "name"

            )

        )

        print(

            "Dataset :",

            self.get(

                "dataset",

                "root"

            )

        )

        print()

        print(

            "Epochs :",

            self.get(

                "training",

                "epochs"

            )

        )

        print(

            "Batch Size :",

            self.get(

                "training",

                "batch_size"

            )

        )

        print(

            "Learning Rate :",

            self.get(

                "training",

                "learning_rate"

            )

        )

        print()

        print("=" * 70)
