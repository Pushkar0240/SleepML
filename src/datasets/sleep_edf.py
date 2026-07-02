"""
SleepTransitionCNN
==================

Sleep-EDF Dataset Loader

Loads PSG recordings together with their corresponding
hypnogram annotations.

Author
------
Pushkar Singh
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import mne

from src.data import Recording
from src.datasets.base_dataset import BaseDataset
from src.logger import logger_factory

logger = logger_factory.get_logger("SleepEDF")


def _recording_key(path: Path) -> str:
    """Return a Sleep-EDF pairing key shared by PSG and hypnogram files."""

    stem = path.stem
    if stem.endswith("-PSG"):
        stem = stem[:-4]
    elif stem.endswith("-Hypnogram"):
        stem = stem[:-10]
    return stem[:-1]


class SleepEDFDataset(BaseDataset):
    """Loader for the Sleep-EDF Expanded dataset."""

    def __init__(self, dataset_root: str | Path):
        super().__init__(dataset_root)
        self.scan()

    def scan(self) -> List[dict]:
        """Scan the dataset root for PSG/Hypnogram pairs."""

        self.recordings.clear()

        if not self.dataset_root.exists():
            logger.warning("Dataset root does not exist: %s", self.dataset_root)
            return self.recordings

        psg_files = sorted(self.dataset_root.rglob("*-PSG.edf"))
        hypnogram_lookup = {
            _recording_key(hyp_path): hyp_path
            for hyp_path in self.dataset_root.rglob("*-Hypnogram.edf")
        }
        logger.info("Found %d PSG files.", len(psg_files))

        for psg in psg_files:
            hyp = hypnogram_lookup.get(_recording_key(psg))
            self.recordings.append(
                {
                    "name": psg.stem,
                    "psg": psg,
                    "hypnogram": hyp,
                }
            )

        return self.recordings

    def summary(self):
        """Log a short dataset summary."""

        logger.info("Sleep-EDF Dataset")
        logger.info("Root : %s", self.dataset_root)
        logger.info("Recordings : %d", len(self.recordings))

    def load(self, index: int) -> Recording:
        """Load one Sleep-EDF recording as a Recording object."""

        if index < 0 or index >= len(self.recordings):
            raise IndexError(f"Recording index out of range: {index}")

        info = self.recordings[index]
        logger.info("Loading %s", info["name"])

        raw = mne.io.read_raw_edf(info["psg"], preload=True, verbose=False)

        annotations = None
        if info["hypnogram"] is not None:
            annotations = mne.read_annotations(info["hypnogram"])
            raw.set_annotations(annotations)

        record = Recording(
            name=info["name"],
            dataset="SleepEDF",
            psg_path=info["psg"],
            hypnogram_path=info["hypnogram"],
        )

        record.raw = raw
        record.annotations = raw.annotations
        record.signal = raw.get_data()
        record.sampling_rate = float(raw.info["sfreq"])
        record.duration = float(raw.times[-1]) if len(raw.times) else 0.0

        if record.annotations is not None:
            record.sleep_stage_labels = [str(stage) for stage in record.annotations.description]
            record.transition_intervals = [
                (float(onset), float(onset + duration), str(description))
                for onset, duration, description in zip(
                    record.annotations.onset,
                    record.annotations.duration,
                    record.annotations.description,
                )
            ]
            record.transition_labels = record.annotations.description.copy()

        for channel in raw.ch_names:
            channel_name = channel.lower()
            if "eeg" in channel_name:
                record.eeg_channels.append(channel)
            elif "eog" in channel_name:
                record.eog_channels.append(channel)
            elif "emg" in channel_name:
                record.emg_channels.append(channel)
            elif "ecg" in channel_name:
                record.ecg_channels.append(channel)
            else:
                record.misc_channels.append(channel)

        logger.info(
            "Loaded recording '%s' with %d channels.",
            record.name,
            record.number_of_channels(),
        )

        return record
