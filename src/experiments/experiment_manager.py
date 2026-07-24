"""
=========================================================
SleepTransitionFramework

Experiment Manager

Creates and manages experiment directories.

Responsibilities
----------------
- Generate experiment IDs
- Create experiment folders
- Save configuration snapshot
- Provide output paths
=========================================================
"""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path


class ExperimentManager:
    """
    Creates a unique workspace for each experiment.
    """

    # ---------------------------------------------------------

    def __init__(

        self,

        output_root="outputs/experiments",

        config_file="config/config.yaml"

    ):

        self.output_root = Path(output_root)

        self.config_file = Path(config_file)

        self.output_root.mkdir(

            parents=True,

            exist_ok=True

        )

        self.experiment_id = self._generate_experiment_id()

        self.experiment_dir = (

            self.output_root /

            self.experiment_id

        )

    # ---------------------------------------------------------

    def _generate_experiment_id(self):

        """
        Example

        experiment_20260704_191530
        """

        timestamp = datetime.now().strftime(

            "%Y%m%d_%H%M%S"

        )

        return f"experiment_{timestamp}"

    # ---------------------------------------------------------

    def create(self):

        """
        Create directory structure.
        """

        folders = [

            "models",

            "logs",

            "history",

            "evaluation",

            "plots",

            "predictions"

        ]

        self.experiment_dir.mkdir(

            parents=True,

            exist_ok=True

        )

        for folder in folders:

            (

                self.experiment_dir /

                folder

            ).mkdir(

                exist_ok=True

            )

        self.save_config()

        return self.experiment_dir

    # ---------------------------------------------------------

    def save_config(self):

        """
        Save configuration snapshot.
        """

        if self.config_file.exists():

            shutil.copy(

                self.config_file,

                self.experiment_dir /

                "config.yaml"

            )

    # ---------------------------------------------------------

    def path(self):

        return self.experiment_dir

    # ---------------------------------------------------------

    def models_dir(self):

        return self.experiment_dir / "models"

    # ---------------------------------------------------------

    def logs_dir(self):

        return self.experiment_dir / "logs"

    # ---------------------------------------------------------

    def history_dir(self):

        return self.experiment_dir / "history"

    # ---------------------------------------------------------

    def evaluation_dir(self):

        return self.experiment_dir / "evaluation"

    # ---------------------------------------------------------

    def plots_dir(self):

        return self.experiment_dir / "plots"

    # ---------------------------------------------------------

    def predictions_dir(self):

        return self.experiment_dir / "predictions"

    # ---------------------------------------------------------

    def summary(self):

        print()

        print("=" * 70)

        print("EXPERIMENT")

        print("=" * 70)

        print()

        print(

            "ID :", self.experiment_id

        )

        print()

        print(

            "Directory :",

            self.experiment_dir

        )

        print()

        print("=" * 70)
