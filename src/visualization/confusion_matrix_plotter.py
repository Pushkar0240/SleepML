"""
=========================================================
SleepTransitionFramework

Confusion Matrix Plotter

Publication-quality confusion matrix visualization.

Responsibilities
----------------
* Compute confusion matrix
* Normalized confusion matrix
* Publication-quality visualization
* Save figure
=========================================================
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import confusion_matrix

from src.visualization.base_plot import BasePlot


class ConfusionMatrixPlotter(BasePlot):

    """
    Plot confusion matrices.
    """

    # ---------------------------------------------------------

    def __init__(

        self,

        output_dir="outputs/evaluation",

        class_names=None

    ):

        super().__init__(

            output_dir=output_dir

        )

        if class_names is None:

            class_names = [

                "Non-Transition",

                "Transition"

            ]

        self.class_names = class_names

        self.cm = None

    # ---------------------------------------------------------

    def compute(

        self,

        y_true,

        y_pred,

        normalize=False

    ):

        """
        Compute confusion matrix.
        """

        self.cm = confusion_matrix(

            y_true,

            y_pred

        )

        if normalize:

            self.cm = (

                self.cm.astype(float)

                /

                self.cm.sum(

                    axis=1,

                    keepdims=True

                )

            )

        return self.cm

    # ---------------------------------------------------------

    def plot(

        self,

        cmap="Blues",

        colorbar=True

    ):

        """
        Plot confusion matrix.
        """

        if self.cm is None:

            raise ValueError(

                "Run compute() first."

            )

        self.create()

        image = self.axis.imshow(

            self.cm,

            cmap=cmap

        )

        if colorbar:

            self.figure.colorbar(

                image,

                ax=self.axis

            )

        self.axis.set_xticks(

            np.arange(

                len(self.class_names)

            )

        )

        self.axis.set_yticks(

            np.arange(

                len(self.class_names)

            )

        )

        self.axis.set_xticklabels(

            self.class_names

        )

        self.axis.set_yticklabels(

            self.class_names

        )

        self.xlabel(

            "Predicted Label"

        )

        self.ylabel(

            "True Label"

        )

        self.title(

            "Confusion Matrix"

        )

        return image

    # ---------------------------------------------------------

    def annotate(

        self,

        decimals=2,

        text_color="black"

    ):

        """
        Write values inside matrix.
        """

        rows, cols = self.cm.shape

        for i in range(rows):

            for j in range(cols):

                value = self.cm[i, j]

                if isinstance(

                    value,

                    float

                ):

                    text = f"{value:.{decimals}f}"

                else:

                    text = str(value)

                self.axis.text(

                    j,

                    i,

                    text,

                    ha="center",

                    va="center",

                    color=text_color,

                    fontsize=11

                )

    # ---------------------------------------------------------
    # Confusion Matrix Statistics
    # ---------------------------------------------------------

    def statistics(self):
        """
        Compute statistics from the confusion matrix.
        """

        if self.cm is None:
            raise ValueError(
                "Confusion matrix has not been computed."
            )

        if self.cm.shape != (2, 2):
            raise ValueError(
                "Statistics currently support binary classification only."
            )

        tn, fp, fn, tp = self.cm.ravel()

        total = tp + tn + fp + fn

        accuracy = (tp + tn) / total if total else 0.0

        precision = tp / (tp + fp) if (tp + fp) else 0.0

        recall = tp / (tp + fn) if (tp + fn) else 0.0

        specificity = tn / (tn + fp) if (tn + fp) else 0.0

        f1 = (
            2 * precision * recall /
            (precision + recall)
            if (precision + recall)
            else 0.0
        )

        return {

            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),

            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "Specificity": specificity,
            "F1 Score": f1

        }

    # ---------------------------------------------------------

    def print_summary(self):
        """
        Print confusion matrix summary.
        """

        stats = self.statistics()

        print()

        print("=" * 70)

        print("CONFUSION MATRIX SUMMARY")

        print("=" * 70)

        print()

        print(

            f"TN : {stats['TN']}"

        )

        print(

            f"FP : {stats['FP']}"

        )

        print(

            f"FN : {stats['FN']}"

        )

        print(

            f"TP : {stats['TP']}"

        )

        print()

        print(

            f"Accuracy   : {stats['Accuracy']:.4f}"

        )

        print(

            f"Precision  : {stats['Precision']:.4f}"

        )

        print(

            f"Recall     : {stats['Recall']:.4f}"

        )

        print(

            f"Specificity: {stats['Specificity']:.4f}"

        )

        print(

            f"F1 Score   : {stats['F1 Score']:.4f}"

        )

        print()

        print("=" * 70)

    # ---------------------------------------------------------

    def export_report(

        self,

        filename="confusion_matrix_report.csv"

    ):
        """
        Export statistics.
        """

        import pandas as pd

        filepath = self.output_dir / filename

        pd.DataFrame(

            [self.statistics()]

        ).to_csv(

            filepath,

            index=False

        )

        print()

        print(

            f"Report exported -> {filepath}"

        )

    # ---------------------------------------------------------

    def save(

        self,

        filename="confusion_matrix"

    ):
        """
        Save figure in multiple formats.
        """

        self.export(

            filename,

            formats=(

                "png",

                "pdf",

                "svg"

            )

        )

        self.close()

    # ---------------------------------------------------------

    def plot_and_save(

        self,

        y_true,

        y_pred,

        normalize=False,

        filename="confusion_matrix"

    ):
        """
        Complete workflow.
        """

        self.compute(

            y_true,

            y_pred,

            normalize=normalize

        )

        self.plot()

        self.annotate()

        self.save(

            filename

        )

        self.export_report()

        self.print_summary()

    # ---------------------------------------------------------

    @property
    def matrix(self):

        return self.cm

    # ---------------------------------------------------------

    def __repr__(self):

        shape = None

        if self.cm is not None:

            shape = self.cm.shape

        return (

            f"{self.__class__.__name__}"

            f"(shape={shape}, "

            f"classes={self.class_names})"

        )
