"""
=========================================================
SleepTransitionFramework

History Plotter

Visualizes model training history.

Responsibilities
----------------
* Loss
* Accuracy
* Precision
* Recall
* AUC
* Learning Rate
* Multi-metric plots
=========================================================
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.visualization.base_plot import BasePlot


class HistoryPlotter(BasePlot):

    """
    Plot TensorFlow training history.
    """

    # ---------------------------------------------------------

    def __init__(

        self,

        history=None,

        history_file=None,

        output_dir="outputs/history"

    ):

        super().__init__(

            output_dir=output_dir

        )

        self.history = None

        if history is not None:

            self.load_history(

                history

            )

        elif history_file is not None:

            self.load_csv(

                history_file

            )

    # ---------------------------------------------------------

    def load_history(

        self,

        history

    ):

        """
        TensorFlow History object.
        """

        if hasattr(

            history,

            "history"

        ):

            self.history = pd.DataFrame(

                history.history

            )

        else:

            self.history = pd.DataFrame(

                history

            )

    # ---------------------------------------------------------

    def load_csv(

        self,

        filename

    ):

        filename = Path(filename)

        self.history = pd.read_csv(

            filename

        )

    # ---------------------------------------------------------

    @property

    def epochs(self):

        return range(

            1,

            len(self.history) + 1

        )

    # ---------------------------------------------------------

    def available_metrics(self):

        return list(

            self.history.columns

        )

    # ---------------------------------------------------------

    def plot_metric(

        self,

        metric,

        validation=True,

        title=None,

        ylabel=None,

        filename=None

    ):

        self.create()

        self.axis.plot(

            self.epochs,

            self.history[metric],

            linewidth=2,

            label=f"Training {metric}"

        )

        val_metric = f"val_{metric}"

        if (

            validation

            and

            val_metric in self.history.columns

        ):

            self.axis.plot(

                self.epochs,

                self.history[val_metric],

                linewidth=2,

                label=f"Validation {metric}"

            )

        self.title(

            title or metric.capitalize()

        )

        self.xlabel(

            "Epoch"

        )

        self.ylabel(

            ylabel or metric.capitalize()

        )

        self.grid()

        self.legend()

        self.integer_x_axis()

        if filename:

            self.export(

                filename,

                formats=(

                    "png",

                    "pdf"

                )

            )

            self.close()

    # ---------------------------------------------------------

    def plot_loss(self):

        self.plot_metric(

            metric="loss",

            title="Training Loss",

            ylabel="Loss",

            filename="loss"

        )

    # ---------------------------------------------------------

    def plot_accuracy(self):

        if "accuracy" not in self.history.columns:

            return

        self.plot_metric(

            metric="accuracy",

            title="Training Accuracy",

            ylabel="Accuracy",

            filename="accuracy"

        )

    # ---------------------------------------------------------

    def plot_precision(self):

        if "precision" not in self.history.columns:

            return

        self.plot_metric(

            metric="precision",

            title="Training Precision",

            ylabel="Precision",

            filename="precision"

        )

    # ---------------------------------------------------------

    def plot_recall(self):

        if "recall" not in self.history.columns:

            return

        self.plot_metric(

            metric="recall",

            title="Training Recall",

            ylabel="Recall",

            filename="recall"

        )

    # ---------------------------------------------------------

    def plot_auc(self):

        """
        Plot AUC curve.
        """

        if "auc" not in self.history.columns:
            return

        self.plot_metric(

            metric="auc",

            title="Training AUC",

            ylabel="AUC",

            filename="auc"

        )

    # ---------------------------------------------------------

    def plot_learning_rate(self):

        """
        Plot learning rate schedule.
        """

        if "learning_rate" not in self.history.columns:

            if "lr" not in self.history.columns:

                return

            column = "lr"

        else:

            column = "learning_rate"

        self.create()

        self.axis.plot(

            self.epochs,

            self.history[column],

            linewidth=2

        )

        self.title(

            "Learning Rate Schedule"

        )

        self.xlabel(

            "Epoch"

        )

        self.ylabel(

            "Learning Rate"

        )

        self.grid()

        self.integer_x_axis()

        self.export(

            "learning_rate",

            formats=("png", "pdf")

        )

        self.close()

    # ---------------------------------------------------------

    def best_epoch(

        self,

        metric="val_loss",

        mode="min"

    ):

        """
        Return best epoch.
        """

        if metric not in self.history.columns:

            return None

        if mode == "min":

            idx = self.history[metric].idxmin()

        else:

            idx = self.history[metric].idxmax()

        return {

            "epoch": idx + 1,

            "value": self.history.loc[idx, metric]

        }

    # ---------------------------------------------------------

    def summary(self):

        """
        Print history summary.
        """

        print()

        print("=" * 70)

        print("TRAINING HISTORY SUMMARY")

        print("=" * 70)

        print()

        print(

            "Epochs :",

            len(self.history)

        )

        print()

        metrics = [

            "loss",

            "accuracy",

            "precision",

            "recall",

            "auc"

        ]

        for metric in metrics:

            if metric in self.history.columns:

                print(

                    f"Final {metric.capitalize()} :",

                    self.history[metric].iloc[-1]

                )

        print()

        if "val_loss" in self.history.columns:

            best = self.best_epoch(

                "val_loss",

                "min"

            )

            print(

                "Best Validation Loss"

            )

            print(

                f"Epoch {best['epoch']}"

            )

            print(

                f"Loss : {best['value']:.6f}"

            )

            print()

        print("=" * 70)

        print()

    # ---------------------------------------------------------

    def dashboard(self):

        """
        Generate every available plot.
        """

        self.plot_loss()

        self.plot_accuracy()

        self.plot_precision()

        self.plot_recall()

        self.plot_auc()

        self.plot_learning_rate()

    # ---------------------------------------------------------

    def export_report(

        self,

        filename="training_history.csv"

    ):

        """
        Export history CSV.
        """

        filepath = self.output_dir / filename

        self.history.to_csv(

            filepath,

            index=False

        )

        print()

        print(

            f"History exported -> {filepath}"

        )

    # ---------------------------------------------------------

    def latest_metrics(self):

        """
        Return latest metrics.
        """

        return self.history.iloc[-1].to_dict()

    # ---------------------------------------------------------

    def metric_exists(

        self,

        metric

    ):

        return metric in self.history.columns

    # ---------------------------------------------------------

    def __len__(self):

        return len(self.history)

    # ---------------------------------------------------------

    def __repr__(self):

        return (

            f"HistoryPlotter("

            f"epochs={len(self)}, "

            f"metrics={len(self.history.columns)})"

        )
