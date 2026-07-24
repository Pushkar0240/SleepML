"""SleepTransitionCNN configuration manager."""

from __future__ import annotations

import random
import subprocess
from pathlib import Path

import numpy as np
import yaml

try:
    import tensorflow as tf
except ModuleNotFoundError:
    tf = None


class Config:
    """Load, validate, and expose project configuration values."""

    def __init__(self, config_path: str | Path = "config.yaml") -> None:
        self.project_root = Path(__file__).resolve().parent.parent
        self.config_path = self.project_root / config_path

        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found:\n{self.config_path}"
            )

        with open(self.config_path, "r", encoding="utf-8") as file:
            self.cfg = yaml.safe_load(file)

        self._validate()
        self._create_directories()
        self.device = self._detect_device()
        self._set_random_seed()

    def _validate(self) -> None:
        required_sections = (
            "project",
            "paths",
            "dataset",
            "preprocessing",
            "window",
            "training",
            "model",
            "logging",
            "visualization",
        )
        missing_sections = [section for section in required_sections if section not in self.cfg]
        if missing_sections:
            missing_text = ", ".join(missing_sections)
            raise ValueError(f"Missing configuration sections: {missing_text}")

    def get(self, *keys):
        """Return a nested configuration value."""

        data = self.cfg
        for key in keys:
            data = data[key]
        return data

    @property
    def paths(self):
        """Return the configured path mapping."""

        return self.cfg["paths"]

    @property
    def public_data(self):
        return self.project_root / self.paths["public_data"]

    @property
    def private_data(self):
        return self.project_root / self.paths["private_data"]

    @property
    def processed_data(self):
        return self.project_root / self.paths["processed_data"]

    @property
    def windows(self):
        return self.project_root / self.paths["windows"]

    @property
    def dataset(self):
        return self.project_root / self.paths["dataset"]

    @property
    def models(self):
        return self.project_root / self.paths["models"]

    @property
    def outputs(self):
        return self.project_root / self.paths["outputs"]

    @property
    def figures(self):
        return self.project_root / self.paths["figures"]

    @property
    def reports(self):
        return self.project_root / self.paths["reports"]

    @property
    def predictions(self):
        return self.project_root / self.paths["predictions"]

    @property
    def logs(self):
        return self.project_root / self.paths["logs"]

    def _create_directories(self) -> None:
        for folder in self.cfg["paths"].values():
            (self.project_root / folder).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _detect_device():
        if tf is not None:
            gpus = tf.config.list_physical_devices("GPU")
            if len(gpus):
                return "GPU"

        try:
            completed_process = subprocess.run(
                ["nvidia-smi"],
                capture_output=True,
                check=False,
                text=True,
            )
            if completed_process.returncode == 0:
                return "GPU"
        except OSError:
            pass

        return "CPU"

    def _set_random_seed(self) -> None:
        seed = self.cfg["project"]["random_seed"]
        random.seed(seed)
        np.random.seed(seed)
        if tf is not None:
            tf.random.set_seed(seed)

    def summary(self):
        """Print a compact summary of the loaded configuration."""

        print("=" * 70)
        print("SleepTransitionCNN Configuration")
        print("=" * 70)
        print(f"Project : {self.get('project', 'name')}")
        print(f"Version : {self.get('project', 'version')}")
        print(f"Device  : {self.device}")
        print()
        print("Directories")
        print("-" * 70)
        print(f"Public Data     : {self.public_data}")
        print(f"Private Data    : {self.private_data}")
        print(f"Processed Data  : {self.processed_data}")
        print(f"Windows         : {self.windows}")
        print(f"Dataset         : {self.dataset}")
        print(f"Models          : {self.models}")
        print(f"Outputs         : {self.outputs}")
        print("=" * 70)


config = Config()