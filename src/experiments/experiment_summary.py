"""
=========================================================
SleepTransitionFramework

Experiment Summary

Generates summaries and rankings of all experiments.

Responsibilities
----------------
- Display experiment history
- Rank experiments
- Find best experiment
- Compute statistics
- Export summary
=========================================================
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class ExperimentSummary:
    """
    Summarize all recorded experiments.
    """

    # ---------------------------------------------------------

    def __init__(

        self,

        registry_path="outputs/experiments/experiment_registry.csv"

    ):

        self.registry_path = Path(registry_path)

        if not self.registry_path.exists():

            raise FileNotFoundError(

                f"Registry not found:\n{self.registry_path}"

            )

        self.df = pd.read_csv(self.registry_path)

    # ---------------------------------------------------------

    def experiments(self):

        return self.df

    # ---------------------------------------------------------

    def total_experiments(self):

        return len(self.df)

    # ---------------------------------------------------------

    def completed(self):

        return self.df[

            self.df["status"] == "Completed"

        ]

    # ---------------------------------------------------------

    def best_accuracy(self):

        completed = self.completed()

        if completed.empty:

            return None

        idx = completed["accuracy"].idxmax()

        return completed.loc[idx]

    # ---------------------------------------------------------

    def best_auc(self):

        completed = self.completed()

        if completed.empty:

            return None

        idx = completed["auc"].idxmax()

        return completed.loc[idx]

    # ---------------------------------------------------------

    def leaderboard(

        self,

        metric="accuracy",

        top_k=10

    ):

        completed = self.completed()

        if completed.empty:

            return completed

        return completed.sort_values(

            metric,

            ascending=False

        ).head(top_k)

    # ---------------------------------------------------------

    def statistics(self):

        completed = self.completed()

        if completed.empty:

            return {}

        return {

            "total_experiments":

                len(self.df),

            "completed":

                len(completed),

            "best_accuracy":

                completed["accuracy"].max(),

            "mean_accuracy":

                completed["accuracy"].mean(),

            "best_auc":

                completed["auc"].max(),

            "mean_auc":

                completed["auc"].mean()

        }

    # ---------------------------------------------------------

    def export(

        self,

        filename="outputs/experiments/summary.csv"

    ):

        leaderboard = self.leaderboard()

        leaderboard.to_csv(

            filename,

            index=False

        )

        print(

            f"Summary exported -> {filename}"

        )

    # ---------------------------------------------------------

    def print_summary(self):

        stats = self.statistics()

        print()

        print("=" * 70)

        print("EXPERIMENT SUMMARY")

        print("=" * 70)

        print()

        print(

            "Total Experiments :",

            stats.get(

                "total_experiments",

                0

            )

        )

        print(

            "Completed :",

            stats.get(

                "completed",

                0

            )

        )

        print()

        print(

            "Best Accuracy :",

            stats.get(

                "best_accuracy",

                "-"

            )

        )

        print(

            "Mean Accuracy :",

            stats.get(

                "mean_accuracy",

                "-"

            )

        )

        print()

        print(

            "Best AUC :",

            stats.get(

                "best_auc",

                "-"

            )

        )

        print(

            "Mean AUC :",

            stats.get(

                "mean_auc",

                "-"

            )

        )

        print()

        print("=" * 70)

    # ---------------------------------------------------------

    def print_leaderboard(

        self,

        metric="accuracy",

        top_k=10

    ):

        board = self.leaderboard(

            metric,

            top_k

        )

        print()

        print("=" * 70)

        print(f"TOP {top_k} EXPERIMENTS")

        print("=" * 70)

        print()

        print(board)

        print()
