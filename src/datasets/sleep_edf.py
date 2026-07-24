"""
SleepTransitionCNN
==================

SleepEDF Dataset Loader

Loads Sleep-EDF PSG and Hypnogram EDF files and
returns a unified Recording object.
"""

from __future__ import annotations

from pathlib import Path

import mne

from src.data.recording import Recording


class SleepEDFDataset:

    def __init__(self, dataset_path: Path):

        self.dataset_path = Path(dataset_path)

        self.psg_files = sorted(

            self.dataset_path.glob("*PSG.edf")

        )

        self.hyp_files = sorted(

            self.dataset_path.glob("*Hypnogram.edf")

        )

        if len(self.psg_files) != len(self.hyp_files):

            print(

                "Warning: Number of PSG and Hypnogram files differs."

            )

    # -----------------------------------------------------

    def __len__(self):

        return len(self.psg_files)

    # -----------------------------------------------------

    def load(self, index: int) -> Recording:

        psg = self.psg_files[index]

        hyp = self.hyp_files[index]

        raw = mne.io.read_raw_edf(

            psg,

            preload=True,

            verbose=False

        )

        annotations = mne.read_annotations(hyp)

        raw.set_annotations(

            annotations

        )

        recording_id = psg.stem

        subject_id = recording_id[:6]

        record = Recording(

            subject_id=subject_id,

            recording_id=recording_id,

            dataset_name="SleepEDF",

            raw=raw

        )

        record.update_from_raw()

        record.add_history(

            "SleepEDF Loaded"

        )

        return record

    # -----------------------------------------------------

    def summary(self):

        print()

        print("=" * 60)

        print("SleepEDF Dataset")

        print("=" * 60)

        print(f"Dataset : {self.dataset_path}")

        print(f"Recordings : {len(self)}")

        print("=" * 60)
