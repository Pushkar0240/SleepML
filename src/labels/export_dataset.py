"""
SleepTransitionCNN
==================

Dataset Exporter

Exports TrainingSample objects as
X.npy and y.npy
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data.training_sample import TrainingSample


class DatasetExporter:

    def __init__(self):

        pass

    # -----------------------------------------------------

    def export(

        self,

        samples: list[TrainingSample],

        output_folder="data/dataset"

    ):

        output = Path(output_folder)

        output.mkdir(

            parents=True,

            exist_ok=True

        )

        X = []

        y = []

        for sample in samples:

            X.append(

                sample.signal

            )

            y.append(

                sample.label

            )

        X = np.asarray(

            X,

            dtype=np.float32

        )

        y = np.asarray(

            y,

            dtype=np.int32

        )

        np.save(

            output / "X.npy",

            X

        )

        np.save(

            output / "y.npy",

            y

        )

        print()

        print("=" * 70)

        print("Dataset Exported")

        print("=" * 70)

        print()

        print("X Shape :", X.shape)

        print("y Shape :", y.shape)

        print()

        print("Saved To")

        print(output)

        print()

        return X, y

    # -----------------------------------------------------

    def load(

        self,

        dataset_folder="data/dataset"

    ):

        dataset = Path(dataset_folder)

        X = np.load(

            dataset / "X.npy"

        )

        y = np.load(

            dataset / "y.npy"

        )

        print()

        print("Dataset Loaded")

        print()

        print("X :", X.shape)

        print("y :", y.shape)

        return X, y
