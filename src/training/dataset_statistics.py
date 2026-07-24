"""
SleepTransitionCNN
==================

Dataset Statistics

Generates statistics for the complete dataset.


"""

from __future__ import annotations

from collections import Counter
import pandas as pd


class DatasetStatistics:

    def __init__(self):

        self.metadata = None

    # ----------------------------------------------------
    # Load Metadata
    # ----------------------------------------------------

    def load(self, metadata):

        if isinstance(metadata, str):

            self.metadata = pd.read_csv(metadata)

        else:

            self.metadata = metadata

    # ----------------------------------------------------
    # Print Report
    # ----------------------------------------------------

    def report(self):

        df = self.metadata

        print("\n" + "=" * 70)
        print("DATASET REPORT")
        print("=" * 70)

        print(f"Total Samples      : {len(df)}")

        print(f"Subjects           : {df['subject_id'].nunique()}")

        print(f"Recordings         : {df['recording_id'].nunique()}")

        print(f"Datasets           : {df['dataset'].unique().tolist()}")

        print("=" * 70)

        print("\nClass Distribution")

        labels = Counter(df["label"])

        for label, count in labels.items():

            print(f"{label:15} : {count}")

        print("\nTransition Types")

        transitions = Counter(df["transition_type"])

        for name, count in transitions.items():

            print(f"{name:25} : {count}")

        print("\nDataset Distribution")

        datasets = Counter(df["dataset"])

        for name, count in datasets.items():

            print(f"{name:15} : {count}")

        print("\nAverage Sampling Rate")

        print(df["sampling_rate"].mean())

        print("=" * 70)

    # ----------------------------------------------------
    # Save Report
    # ----------------------------------------------------

    def export(self, output="outputs/dataset_report.txt"):

        with open(output, "w") as file:

            df = self.metadata

            file.write("DATASET REPORT\n")

            file.write("=" * 50 + "\n")

            file.write(f"Samples : {len(df)}\n")

            file.write(

                f"Subjects : {df['subject_id'].nunique()}\n"

            )

            file.write(

                f"Recordings : {df['recording_id'].nunique()}\n"

            )

            file.write(

                f"Sampling Rate : {df['sampling_rate'].mean()}\n"

            )

        print("\nDataset report saved:", output)
