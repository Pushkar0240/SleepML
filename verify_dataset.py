"""
=========================================================
SleepTransitionCNN

Dataset Verification Tool

Checks whether the generated dataset is valid
before training the CNN.
=========================================================
"""

from pathlib import Path
import numpy as np
import pandas as pd


class DatasetVerifier:

    def __init__(self):

        self.dataset_dir = Path("data/dataset")

        self.errors = []

        self.warnings = []

    # ---------------------------------------------------
    # Helper
    # ---------------------------------------------------

    def ok(self, message):

        print(f"[PASS] {message}")

    def warning(self, message):

        print(f"[WARNING] {message}")

        self.warnings.append(message)

    def error(self, message):

        print(f"[FAILED] {message}")

        self.errors.append(message)

    # ---------------------------------------------------
    # Check Files
    # ---------------------------------------------------

    def check_files(self):

        print("\nChecking required files...\n")

        required = [

            "X.npy",

            "y.npy",

            "metadata.csv"

        ]

        for file in required:

            path = self.dataset_dir / file

            if path.exists():

                self.ok(file)

            else:

                self.error(f"{file} missing")

    # ---------------------------------------------------
    # Load Dataset
    # ---------------------------------------------------

    def load(self):

        self.X = np.load(

            self.dataset_dir / "X.npy",

            allow_pickle=True

        )

        self.y = np.load(

            self.dataset_dir / "y.npy"

        )

        self.meta = pd.read_csv(

            self.dataset_dir / "metadata.csv"

        )

    # ---------------------------------------------------
    # Shapes
    # ---------------------------------------------------

    def check_shapes(self):

        print("\nChecking dataset shape...\n")

        print("X :", self.X.shape)

        print("y :", self.y.shape)

        print("Metadata :", len(self.meta))

        if len(self.X) == len(self.y):

            self.ok("X and y length match")

        else:

            self.error("X and y mismatch")

        if len(self.X) == len(self.meta):

            self.ok("Metadata rows match")

        else:

            self.error("Metadata mismatch")

    # ---------------------------------------------------
    # Labels
    # ---------------------------------------------------

    def check_labels(self):

        print("\nChecking labels...\n")

        unique = np.unique(self.y)

        print("Labels :", unique)

        if len(unique) < 2:

            self.warning(

                "Only one class found."

            )

        else:

            self.ok("Multiple classes detected")

    # ---------------------------------------------------
    # NaN
    # ---------------------------------------------------

    def check_nan(self):

        print("\nChecking NaN values...\n")

        if np.isnan(self.X).any():

            self.error("NaN found in X")

        else:

            self.ok("No NaN in X")

        if np.isnan(self.y).any():

            self.error("NaN found in y")

        else:

            self.ok("No NaN in y")

    # ---------------------------------------------------
    # Window Size
    # ---------------------------------------------------

    def check_window_size(self):

        print("\nChecking window sizes...\n")

        lengths = [

            sample.shape[-1]

            for sample in self.X

        ]

        unique = set(lengths)

        if len(unique) == 1:

            self.ok(

                f"All windows = {unique.pop()} samples"

            )

        else:

            self.error("Window size mismatch")

    # ---------------------------------------------------
    # Metadata
    # ---------------------------------------------------

    def check_metadata(self):

        print("\nChecking metadata...\n")

        required = [

            "sample_id",

            "subject_id",

            "recording_id",

            "dataset",

            "label"

        ]

        for column in required:

            if column in self.meta.columns:

                self.ok(column)

            else:

                self.error(f"{column} missing")

    # ---------------------------------------------------
    # Summary
    # ---------------------------------------------------

    def summary(self):

        print("\n")

        print("=" * 70)

        print("DATASET VERIFICATION")

        print("=" * 70)

        print()

        print("Errors :", len(self.errors))

        print("Warnings :", len(self.warnings))

        print()

        if len(self.errors) == 0:

            print("PROJECT STATUS : PASS")

        else:

            print("PROJECT STATUS : FAILED")

        print()

        print("=" * 70)

    # ---------------------------------------------------
    # Run Everything
    # ---------------------------------------------------

    def run(self):

        self.check_files()

        if len(self.errors):

            self.summary()

            return

        self.load()

        self.check_shapes()

        self.check_labels()

        self.check_nan()

        self.check_window_size()

        self.check_metadata()

        self.summary()


if __name__ == "__main__":

    DatasetVerifier().run()
