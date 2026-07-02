"""Backward-compatible Sleep-EDF loader entry point."""

from __future__ import annotations

from pathlib import Path

from src.datasets.sleep_edf import SleepEDFDataset


class EDFLoader:
    """Compatibility wrapper around the SleepEDF dataset loader."""

    def __init__(self, edf_path: Path) -> None:
        self.edf_path = Path(edf_path)
        self.record = None

    def load(self):
        dataset = SleepEDFDataset(self.edf_path.parent)

        for index, info in enumerate(dataset.recordings):
            if info["psg"] == self.edf_path:
                self.record = dataset.load(index)
                return self.record.raw

        raise FileNotFoundError(
            f"No matching Sleep-EDF PSG file was found for {self.edf_path}"
        )
