"""
=========================================================
SleepTransitionFramework

Configuration Validator

Validates the project configuration before training.
=========================================================
"""

from __future__ import annotations

from pathlib import Path

from src.config.config_loader import ConfigLoader
from src.models import ModelFactory


class ConfigValidator:

    """
    Validates project configuration.
    """

    def __init__(self, config: ConfigLoader):

        self.config = config

        self.errors = []

        self.warnings = []

    # ---------------------------------------------------------

    def validate_project(self):

        if not self.config.contains("project", "name"):

            self.errors.append("Missing project.name")

        if not self.config.contains("project", "version"):

            self.errors.append("Missing project.version")

    # ---------------------------------------------------------

    def validate_dataset(self):

        dataset_root = Path(

            self.config.get("dataset", "root")

        )

        if not dataset_root.exists():

            self.errors.append(

                f"Dataset folder does not exist:\n{dataset_root}"

            )

    # ---------------------------------------------------------

    def validate_model(self):

        model_name = self.config.get(

            "model",

            "name"

        )

        available = ModelFactory.available_models()

        if model_name not in available:

            self.errors.append(

                f"Unknown model '{model_name}'.\n"

                f"Available models: {available}"

            )

    # ---------------------------------------------------------

    def validate_training(self):

        epochs = self.config.get(

            "training",

            "epochs"

        )

        if epochs <= 0:

            self.errors.append(

                "Epochs must be greater than zero."

            )

        batch = self.config.get(

            "training",

            "batch_size"

        )

        if batch <= 0:

            self.errors.append(

                "Batch size must be greater than zero."

            )

        lr = self.config.get(

            "training",

            "learning_rate"

        )

        if lr <= 0:

            self.errors.append(

                "Learning rate must be positive."

            )

    # ---------------------------------------------------------

    def validate_model_parameters(self):

        dropouts = self.config.get(

            "model",

            "dropout"

        )

        for value in dropouts:

            if value < 0 or value >= 1:

                self.errors.append(

                    f"Invalid dropout value: {value}"

                )

        kernels = self.config.get(

            "model",

            "kernels"

        )

        for kernel in kernels:

            if kernel % 2 == 0:

                self.warnings.append(

                    f"Kernel size {kernel} is even. Odd kernel sizes are recommended."

                )

    # ---------------------------------------------------------

    def validate_output(self):

        output = Path(

            self.config.get(

                "output",

                "root"

            )

        )

        output.mkdir(

            parents=True,

            exist_ok=True

        )

    # ---------------------------------------------------------

    def validate(self):

        self.validate_project()

        self.validate_dataset()

        self.validate_model()

        self.validate_training()

        self.validate_model_parameters()

        self.validate_output()

        self.report()

    # ---------------------------------------------------------

    def report(self):

        print()

        print("=" * 70)

        print("CONFIGURATION VALIDATION")

        print("=" * 70)

        print()

        if self.warnings:

            print("Warnings")

            print("-" * 70)

            for warning in self.warnings:

                print("[WARNING]", warning)

            print()

        if self.errors:

            print("Errors")

            print("-" * 70)

            for error in self.errors:

                print("[ERROR]", error)

            print()

            raise ValueError(

                "Configuration validation failed."

            )

        print("Configuration validation passed.")

        print()
