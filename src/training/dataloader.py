"""
SleepTransitionCNN
==================

TensorFlow Data Loader

Loads the exported NumPy dataset and prepares
tf.data.Dataset pipelines.


"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf


class EEGDataLoader:

    def __init__(

        self,

        dataset_folder=None,

        batch_size=None,

        shuffle=True

    ):

        from src.config import ConfigManager
        config = ConfigManager()
        
        if dataset_folder is None:
            dataset_folder = config.dataset["root"]
            
        if batch_size is None:
            batch_size = config.training["batch_size"]

        self.dataset_folder = Path(dataset_folder)

        self.batch_size = batch_size

        self.shuffle = shuffle

    # -----------------------------------------------------

    def load_numpy(self):

        X = np.load(

            self.dataset_folder / "X.npy"

        )

        y = np.load(

            self.dataset_folder / "y.npy"

        )

        print()

        print("=" * 70)

        print("Dataset Loaded")

        print("=" * 70)

        print("Original Shape :", X.shape)

        print()

        # ------------------------------------------
        # CNN expects:
        #
        # (samples,time_steps,channels)
        #
        # Current:
        #
        # (samples,channels,time)
        # ------------------------------------------

        X = np.transpose(

            X,

            (0, 2, 1)

        )

        print(

            "TensorFlow Shape :",

            X.shape

        )

        print()

        return X, y

    # -----------------------------------------------------

    def dataset(self):

        X, y = self.load_numpy()

        ds = tf.data.Dataset.from_tensor_slices(

            (X, y)

        )

        if self.shuffle:

            ds = ds.shuffle(

                len(X),

                reshuffle_each_iteration=True

            )

        ds = ds.batch(

            self.batch_size

        )

        ds = ds.prefetch(

            tf.data.AUTOTUNE

        )

        return ds

    # -----------------------------------------------------

    def sample(self):

        X, y = self.load_numpy()

        print()

        print("One Sample")

        print()

        print("Signal :", X[0].shape)

        print("Label :", y[0])

        return X[0], y[0]
