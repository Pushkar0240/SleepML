"""
SleepTransitionCNN
==================

Subject-wise Dataset Splitter

Uses metadata.csv to create train, validation,
and test datasets without subject leakage.


"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


class DatasetSplitter:

    def __init__(
        self,
        config: dict = None,
        dataset_folder="data/dataset",
        train_ratio=0.70,
        validation_ratio=0.15,
        test_ratio=0.15,
        random_seed=42
    ):
        self.dataset_folder = Path(dataset_folder)
        
        if config is not None and "training" in config:
            train_cfg = config["training"]
            self.train_ratio = train_cfg.get("train_ratio", train_ratio)
            self.validation_ratio = train_cfg.get("validation_ratio", validation_ratio)
            self.test_ratio = train_cfg.get("test_ratio", test_ratio)
            self.random_seed = train_cfg.get("random_seed", random_seed)
        else:
            self.train_ratio = train_ratio
            self.validation_ratio = validation_ratio
            self.test_ratio = test_ratio
            self.random_seed = random_seed

    # -----------------------------------------------------

    def load(self):

        self.X = np.load(
            self.dataset_folder / "X.npy"
        )

        self.y = np.load(
            self.dataset_folder / "y.npy"
        )

        self.metadata = pd.read_csv(
            self.dataset_folder / "metadata.csv"
        )

    # -----------------------------------------------------

    def split_subjects(self):

        subjects = sorted(
            self.metadata["subject_id"].unique()
        )

        rng = np.random.default_rng(
            self.random_seed
        )

        rng.shuffle(subjects)

        n = len(subjects)

        train_end = int(
            n * self.train_ratio
        )

        val_end = train_end + int(
            n * self.validation_ratio
        )

        train_subjects = subjects[:train_end]

        validation_subjects = subjects[train_end:val_end]

        test_subjects = subjects[val_end:]

        return (
            train_subjects,
            validation_subjects,
            test_subjects
        )

    # -----------------------------------------------------

    def save_split(
        self,
        name,
        indices
    ):

        folder = self.dataset_folder / name

        folder.mkdir(
            parents=True,
            exist_ok=True
        )

        np.save(
            folder / "X.npy",
            self.X[indices]
        )

        np.save(
            folder / "y.npy",
            self.y[indices]
        )

        self.metadata.iloc[indices].to_csv(
            folder / "metadata.csv",
            index=False
        )

        print()
        print(f"{name.upper()}")
        print(
            "Samples :",
            len(indices)
        )

    # -----------------------------------------------------

    def split(self):

        self.load()

        train_subjects, val_subjects, test_subjects = (
            self.split_subjects()
        )

        subjects = self.metadata["subject_id"].unique()
        if len(subjects) < 2:
            print("Warning: Not enough subjects for subject-wise split. Falling back to sample-wise split.")
            indices = np.arange(len(self.metadata))
            np.random.shuffle(indices)
            n_samples = len(indices)
            
            train_end = int(n_samples * self.train_ratio)
            val_end = train_end + int(n_samples * self.validation_ratio)
            
            train_idx = indices[:train_end]
            val_idx = indices[train_end:val_end]
            test_idx = indices[val_end:]
            
            train_subjects = subjects
            val_subjects = subjects
            test_subjects = subjects
        else:
            train_idx = self.metadata[self.metadata.subject_id.isin(train_subjects)].index
            val_idx = self.metadata[self.metadata.subject_id.isin(val_subjects)].index
            test_idx = self.metadata[self.metadata.subject_id.isin(test_subjects)].index

        self.save_split(
            "train",
            train_idx
        )

        self.save_split(
            "validation",
            val_idx
        )

        self.save_split(
            "test",
            test_idx
        )

        print()
        print("=" * 70)
        print("Subject-wise Split Complete")
        print("=" * 70)
        print()

        print(
            "Train Subjects :",
            len(train_subjects)
        )

        print(
            "Validation Subjects :",
            len(val_subjects)
        )

        print(
            "Test Subjects :",
            len(test_subjects)
        )


if __name__ == "__main__":
    splitter = DatasetSplitter()
    splitter.split()
