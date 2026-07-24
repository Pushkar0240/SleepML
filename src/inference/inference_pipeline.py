"""
=========================================================
SleepTransitionFramework

Inference Pipeline

Runs inference on an entire dataset and exports results.
=========================================================
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.inference.predictor import Predictor


class InferencePipeline:
    """
    Complete inference pipeline.

    Responsibilities
    ----------------
    - Load trained model
    - Run inference on all samples
    - Save predictions
    - Generate summary
    """

    # -----------------------------------------------------

    def __init__(

        self,

        predictor: Predictor,

        output_dir="outputs/predictions"

    ):

        self.predictor = predictor

        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(

            parents=True,

            exist_ok=True

        )

    # -----------------------------------------------------

    def run(

        self,

        X,

        metadata=None

    ):

        """
        Run inference.

        Parameters
        ----------
        X : ndarray
            Shape (N, time, channels)

        metadata : DataFrame | None
        """

        results = self.predictor.predict_batch(X)

        rows = []

        for i, result in enumerate(results):

            row = result.to_dict()

            if metadata is not None:

                meta = metadata.iloc[i]

                for column in metadata.columns:

                    row[column] = meta[column]

            rows.append(row)

        df = pd.DataFrame(rows)

        self.results = df

        return df

    # -----------------------------------------------------

    def save(

        self,

        filename="predictions.csv"

    ):

        filepath = self.output_dir / filename

        self.results.to_csv(

            filepath,

            index=False

        )

        print()

        print(f"Predictions saved -> {filepath}")

    # -----------------------------------------------------

    def statistics(self):

        df = self.results

        total = len(df)

        transitions = (

            df["predicted_label"] == 1

        ).sum()

        non_transitions = total - transitions

        confidence = df["confidence"].mean()

        stats = {

            "total_samples": total,

            "transition_predictions": int(transitions),

            "non_transition_predictions": int(non_transitions),

            "mean_confidence": float(confidence)

        }

        return stats

    # -----------------------------------------------------

    def print_summary(self):

        stats = self.statistics()

        print()

        print("=" * 70)

        print("INFERENCE SUMMARY")

        print("=" * 70)

        print()

        print(

            "Samples :", stats["total_samples"]

        )

        print(

            "Transitions :",

            stats["transition_predictions"]

        )

        print(

            "Non-Transitions :",

            stats["non_transition_predictions"]

        )

        print()

        print(

            "Mean Confidence :",

            f"{stats['mean_confidence']:.4f}"

        )

        print()

        print("=" * 70)

    # -----------------------------------------------------

    def export_summary(

        self,

        filename="summary.csv"

    ):

        stats = self.statistics()

        df = pd.DataFrame([stats])

        filepath = self.output_dir / filename

        df.to_csv(

            filepath,

            index=False

        )

        print(

            f"Summary exported -> {filepath}"

        )
