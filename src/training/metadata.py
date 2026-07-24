"""
SleepTransitionCNN
==================

Metadata Exporter

Creates metadata.csv directly from TrainingSample objects.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import pandas as pd

from src.data.training_sample import TrainingSample


class MetadataManager:

    def __init__(self):

        self.metadata = None

    # -----------------------------------------------------

    def build(self, samples: list[TrainingSample]):

        rows = []

        for index, sample in enumerate(samples, start=1):

            rows.append(

                {

                    "sample_id": index,

                    "subject_id": sample.subject_id,

                    "recording_id": sample.recording_id,

                    "dataset": sample.dataset_name,

                    "label": sample.label,

                    "transition_type": sample.transition_type,

                    "previous_stage": sample.previous_stage,

                    "next_stage": sample.next_stage,

                    "start_second": sample.start_second,

                    "center_second": sample.center_second,

                    "end_second": sample.end_second,

                    "duration": sample.duration,

                    "channels": sample.n_channels,

                    "samples": sample.n_samples

                }

            )

        self.metadata = pd.DataFrame(rows)

        return self.metadata

    # -----------------------------------------------------

    def export(

        self,

        samples,

        output_folder="data/dataset"

    ):

        df = self.build(samples)

        output = Path(output_folder)

        output.mkdir(

            parents=True,

            exist_ok=True

        )

        path = output / "metadata.csv"

        df.to_csv(

            path,

            index=False

        )

        print()

        print("=" * 60)

        print("Metadata Exported")

        print(path)

        print(f"Samples : {len(df)}")

        print("=" * 60)

        return path

    # -----------------------------------------------------

    def summary(self):

        if self.metadata is None:

            print("Metadata not built.")

            return

        print()

        print("=" * 60)

        print("Metadata Summary")

        print("=" * 60)

        print(self.metadata.head())

        print()

        print(self.metadata.describe(include="all"))
