"""
=========================================================
SleepTransitionFramework

ROC Curve Plotter

Publication-quality ROC curve visualization.

Responsibilities
----------------
* Compute ROC Curve
* Compute AUC
* Plot ROC Curve
* Compare multiple ROC curves
=========================================================
"""

from __future__ import annotations

import numpy as np

from sklearn.metrics import roc_curve
from sklearn.metrics import auc

from src.visualization.base_plot import BasePlot


class ROCPlotter(BasePlot):

    """
    Receiver Operating Characteristic Plotter.
    """

    # ---------------------------------------------------------

    def __init__(

        self,

        output_dir="outputs/evaluation"

    ):

        super().__init__(

            output_dir=output_dir

        )

        self.models = []

    # ---------------------------------------------------------

    def compute(

        self,

        y_true,

        probabilities

    ):

        """
        Compute ROC curve.
        """

        fpr, tpr, thresholds = roc_curve(

            y_true,

            probabilities

        )

        roc_auc = auc(

            fpr,

            tpr

        )

        return {

            "fpr": fpr,

            "tpr": tpr,

            "thresholds": thresholds,

            "auc": roc_auc

        }

    # ---------------------------------------------------------

    def add_model(

        self,

        y_true,

        probabilities,

        label="Model"

    ):

        """
        Add model ROC.

        Multiple models can be compared
        on one graph.
        """

        roc = self.compute(

            y_true,

            probabilities

        )

        roc["label"] = label

        self.models.append(

            roc

        )

    # ---------------------------------------------------------

    def plot(

        self,

        diagonal=True

    ):

        """
        Plot all ROC curves.
        """

        self.create()

        for model in self.models:

            self.axis.plot(

                model["fpr"],

                model["tpr"],

                linewidth=2,

                label=(

                    f"{model['label']} "

                    f"(AUC={model['auc']:.4f})"

                )

            )

        if diagonal:

            self.axis.plot(

                [0, 1],

                [0, 1],

                linestyle="--",

                linewidth=1.5,

                color="black",

                label="Random"

            )

        self.title(

            "Receiver Operating Characteristic"

        )

        self.xlabel(

            "False Positive Rate"

        )

        self.ylabel(

            "True Positive Rate"

        )

        self.grid()

        self.legend()

    # ---------------------------------------------------------

    def best_model(self):

        """
        Return best ROC.
        """

        if len(self.models) == 0:

            return None

        return max(

            self.models,

            key=lambda x: x["auc"]

        )

    # ---------------------------------------------------------

    def auc_scores(self):

        """
        Return AUC scores.
        """

        scores = {}

        for model in self.models:

            scores[

                model["label"]

            ] = model["auc"]

        return scores

    # ---------------------------------------------------------
    # Best Threshold
    # ---------------------------------------------------------

    def best_threshold(

        self,

        model_index=0

    ):
        """
        Compute the optimal threshold using
        Youden's J statistic.
        """

        if len(self.models) == 0:

            return None

        model = self.models[model_index]

        j_scores = (

            model["tpr"]

            -

            model["fpr"]

        )

        idx = np.argmax(

            j_scores

        )

        return {

            "threshold":

                float(

                    model["thresholds"][idx]

                ),

            "sensitivity":

                float(

                    model["tpr"][idx]

                ),

            "specificity":

                float(

                    1 -

                    model["fpr"][idx]

                )

        }

    # ---------------------------------------------------------

    def export_scores(

        self,

        filename="roc_scores.csv"

    ):

        """
        Export ROC scores.
        """

        import pandas as pd

        rows = []

        for model in self.models:

            rows.append({

                "Model":

                    model["label"],

                "AUC":

                    model["auc"]

            })

        filepath = (

            self.output_dir /

            filename

        )

        pd.DataFrame(

            rows

        ).to_csv(

            filepath,

            index=False

        )

        print()

        print(

            f"ROC report saved -> {filepath}"

        )

    # ---------------------------------------------------------

    def save(

        self,

        filename="roc_curve"

    ):

        """
        Save ROC figure.
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

    def summary(self):

        """
        Print ROC summary.
        """

        print()

        print("=" * 70)

        print("ROC SUMMARY")

        print("=" * 70)

        print()

        if len(self.models) == 0:

            print(

                "No ROC curves available."

            )

            return

        for model in self.models:

            print(

                f"{model['label']:20s}"

                f"AUC = "

                f"{model['auc']:.4f}"

            )

        print()

        best = self.best_model()

        print(

            "Best Model"

        )

        print(

            f"{best['label']}"

        )

        print(

            f"AUC = {best['auc']:.4f}"

        )

        print()

        print("=" * 70)

    # ---------------------------------------------------------

    def plot_and_save(

        self,

        filename="roc_curve"

    ):

        """
        Complete workflow.
        """

        self.plot()

        self.save(

            filename

        )

        self.export_scores()

        self.summary()

    # ---------------------------------------------------------

    def reset(self):

        """
        Clear stored models.
        """

        self.models.clear()

    # ---------------------------------------------------------

    def __len__(self):

        return len(

            self.models

        )

    # ---------------------------------------------------------

    @property

    def number_of_models(self):

        return len(

            self.models

        )

    # ---------------------------------------------------------

    def __repr__(self):

        return (

            f"{self.__class__.__name__}"

            f"(models={len(self.models)})"

        )
