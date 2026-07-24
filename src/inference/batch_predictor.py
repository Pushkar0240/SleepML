"""
=========================================================
SleepTransitionFramework

Batch Predictor

Runs inference on multiple recordings or datasets.

Responsibilities
----------------
- Batch inference
- Folder-based prediction
- Export predictions
- Aggregate statistics
=========================================================
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.inference.predictor import Predictor


class BatchPredictor:
    """
    Predict multiple datasets or recordings.
    """

    # ---------------------------------------------------------

    def __init__(

        self,

        predictor: Predictor

    ):

        self.predictor = predictor

        self.results = []

    # ---------------------------------------------------------

    def predict_array(

        self,

        X,

        metadata=None

    ):
        """
        Predict a NumPy array.

        Parameters
        ----------
        X : ndarray
            Shape (N, time, channels)

        metadata : DataFrame | None
        """

        predictions = self.predictor.predict_batch(X)

        rows = []

        for i, prediction in enumerate(predictions):

            row = prediction.to_dict()

            if metadata is not None:

                meta = metadata.iloc[i]

                for column in metadata.columns:

                    row[column] = meta[column]

            rows.append(row)

        df = pd.DataFrame(rows)

        self.results.append(df)

        return df

    # ---------------------------------------------------------

    def predict_folder(

        self,

        folder

    ):
        """
        Predict all .npy files in a folder.

        Expected structure

        folder/

            subject01.npy

            subject02.npy

            ...

        """

        folder = Path(folder)

        outputs = []

        for file in sorted(folder.glob("*.npy")):

            print(f"Predicting {file.name}")

            X = np.load(file)

            df = self.predict_array(X)

            df["source_file"] = file.name

            outputs.append(df)

        if outputs:

            merged = pd.concat(

                outputs,

                ignore_index=True

            )

        else:

            merged = pd.DataFrame()

        self.results.append(merged)

        return merged

    # ---------------------------------------------------------

    def merge_results(self):

        if len(self.results) == 0:

            return pd.DataFrame()

        return pd.concat(

            self.results,

            ignore_index=True

        )

    # ---------------------------------------------------------

    def save(

        self,

        filename="outputs/predictions/batch_predictions.csv"

    ):

        filename = Path(filename)

        filename.parent.mkdir(

            parents=True,

            exist_ok=True

        )

        merged = self.merge_results()

        merged.to_csv(

            filename,

            index=False

        )

        print()

        print(f"Saved predictions -> {filename}")

    # ---------------------------------------------------------

    def statistics(self):

        merged = self.merge_results()

        if merged.empty:

            return {}

        total = len(merged)

        transitions = (

            merged["predicted_label"] == 1

        ).sum()

        confidence = merged["confidence"].mean()

        return {

            "samples": total,

            "transitions": int(transitions),

            "non_transitions": int(

                total - transitions

            ),

            "mean_confidence": float(confidence)

        }

    # ---------------------------------------------------------

    def print_summary(self):

        stats = self.statistics()

        print()

        print("=" * 70)

        print("BATCH PREDICTION SUMMARY")

        print("=" * 70)

        print()

        if not stats:

            print("No predictions available.")

            return

        print(

            "Samples :",

            stats["samples"]

        )

        print(

            "Transitions :",

            stats["transitions"]

        )

        print(

            "Non-Transitions :",

            stats["non_transitions"]

        )

        print()

        print(

            "Mean Confidence :",

            f"{stats['mean_confidence']:.4f}"

        )

        print()

        print("=" * 70)
