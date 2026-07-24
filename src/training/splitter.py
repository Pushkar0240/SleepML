"""
SleepTransitionCNN
==================

Subject-wise Dataset Splitter

Creates train, validation and test splits
based on subject IDs instead of individual windows.


"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd


class SubjectSplitter:
    """
    Subject-wise dataset splitter.
    """

    def __init__(
        self,
        train_ratio=0.70,
        validation_ratio=0.15,
        test_ratio=0.15,
        random_seed=42
    ):

        assert abs(
            train_ratio +
            validation_ratio +
            test_ratio - 1.0
        ) < 1e-6

        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio
        self.test_ratio = test_ratio

        self.seed = random_seed

    # ---------------------------------------------------------
    # Split Subjects
    # ---------------------------------------------------------

    def split(self, metadata: pd.DataFrame):

        subjects = sorted(

            metadata["subject_id"].unique()

        )

        random.seed(self.seed)

        random.shuffle(subjects)

        n = len(subjects)

        train_end = int(n * self.train_ratio)

        val_end = train_end + int(
            n * self.validation_ratio
        )

        train_subjects = subjects[:train_end]

        validation_subjects = subjects[
            train_end:val_end
        ]

        test_subjects = subjects[val_end:]

        train_df = metadata[
            metadata.subject_id.isin(
                train_subjects
            )
        ]

        validation_df = metadata[
            metadata.subject_id.isin(
                validation_subjects
            )
        ]

        test_df = metadata[
            metadata.subject_id.isin(
                test_subjects
            )
        ]

        return (

            train_df,

            validation_df,

            test_df

        )

    # ---------------------------------------------------------
    # Export Split
    # ---------------------------------------------------------

    def export(

        self,

        train_df,

        validation_df,

        test_df,

        output_folder="data/dataset"

    ):

        output = Path(output_folder)

        output.mkdir(

            parents=True,

            exist_ok=True

        )

        train_df.to_csv(

            output / "train_metadata.csv",

            index=False

        )

        validation_df.to_csv(

            output / "validation_metadata.csv",

            index=False

        )

        test_df.to_csv(

            output / "test_metadata.csv",

            index=False

        )

        split = {

            "train_subjects":

                sorted(

                    train_df.subject_id.unique()

                ),

            "validation_subjects":

                sorted(

                    validation_df.subject_id.unique()

                ),

            "test_subjects":

                sorted(

                    test_df.subject_id.unique()

                )

        }

        with open(

            output / "splits.json",

            "w",

            encoding="utf-8"

        ) as file:

            json.dump(

                split,

                file,

                indent=4

            )

        print()

        print("=" * 70)

        print("Subject Split Completed")

        print("=" * 70)

        print()

        print(

            "Training Subjects :",

            len(split["train_subjects"])

        )

        print(

            "Validation Subjects :",

            len(split["validation_subjects"])

        )

        print(

            "Testing Subjects :",

            len(split["test_subjects"])

        )

        print()

        print("=" * 70)

        return split
