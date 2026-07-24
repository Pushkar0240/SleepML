"""
=========================================================
SleepTransitionFramework

Experiment Tracker

Maintains a registry of all experiments.

Responsibilities
----------------
- Register experiments
- Update experiment status
- Store final metrics
- Save experiment history
=========================================================
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd


class ExperimentTracker:
    """
    Tracks all experiments in a CSV registry.
    """

    # ---------------------------------------------------------

    def __init__(

        self,

        registry_path="outputs/experiments/experiment_registry.csv"

    ):

        self.registry_path = Path(registry_path)

        self.registry_path.parent.mkdir(

            parents=True,

            exist_ok=True

        )

        if not self.registry_path.exists():

            self._create_registry()

    # ---------------------------------------------------------

    def _create_registry(self):

        columns = [

            "experiment_id",

            "date",

            "model",

            "dataset",

            "epochs",

            "batch_size",

            "learning_rate",

            "status",

            "accuracy",

            "precision",

            "recall",

            "f1_score",

            "auc"

        ]

        pd.DataFrame(

            columns=columns

        ).to_csv(

            self.registry_path,

            index=False

        )

    # ---------------------------------------------------------

    def register(

        self,

        experiment_id,

        config

    ):

        df = pd.read_csv(

            self.registry_path

        )

        row = {

            "experiment_id":

                experiment_id,

            "date":

                datetime.now().strftime(

                    "%Y-%m-%d %H:%M:%S"

                ),

            "model":

                config.model["name"],

            "dataset":

                config.dataset["root"],

            "epochs":

                config.training["epochs"],

            "batch_size":

                config.training["batch_size"],

            "learning_rate":

                config.training["learning_rate"],

            "status":

                "Running",

            "accuracy":

                None,

            "precision":

                None,

            "recall":

                None,

            "f1_score":

                None,

            "auc":

                None

        }

        df.loc[len(df)] = row

        df.to_csv(

            self.registry_path,

            index=False

        )

    # ---------------------------------------------------------

    def update_metrics(

        self,

        experiment_id,

        accuracy,

        precision,

        recall,

        f1,

        auc

    ):

        df = pd.read_csv(

            self.registry_path

        )

        idx = df[

            df["experiment_id"] == experiment_id

        ].index

        if len(idx) == 0:

            return

        idx = idx[0]

        df.loc[idx, "accuracy"] = accuracy

        df.loc[idx, "precision"] = precision

        df.loc[idx, "recall"] = recall

        df.loc[idx, "f1_score"] = f1

        df.loc[idx, "auc"] = auc

        df.to_csv(

            self.registry_path,

            index=False

        )

    # ---------------------------------------------------------

    def update_status(

        self,

        experiment_id,

        status

    ):

        df = pd.read_csv(

            self.registry_path

        )

        idx = df[

            df["experiment_id"] == experiment_id

        ].index

        if len(idx) == 0:

            return

        df.loc[idx, "status"] = status

        df.to_csv(

            self.registry_path,

            index=False

        )

    # ---------------------------------------------------------

    def history(self):

        return pd.read_csv(

            self.registry_path

        )

    # ---------------------------------------------------------

    def latest(self):

        df = self.history()

        if len(df) == 0:

            return None

        return df.iloc[-1]

    # ---------------------------------------------------------

    def summary(self):

        df = self.history()

        print()

        print("=" * 70)

        print("EXPERIMENT HISTORY")

        print("=" * 70)

        print()

        print(df)

        print()
