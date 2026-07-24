"""SleepTransitionCNN
==================
Base Dataset Class

Every supported dataset (Sleep-EDF, MASS, ISRUC,
Private EEG) inherits from this class.


"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List

import mne


class BaseDataset(ABC):
    """Abstract dataset class.

    All datasets must implement the same interface.
    """

    def __init__(self, dataset_root: str | Path):
        self.dataset_root = Path(dataset_root)
        self.recordings: List[Dict] = []

    # -----------------------------------------------------
    # Abstract Methods
    # -----------------------------------------------------
    @abstractmethod
    def scan(self) -> List[Dict]:
        """Scan dataset directory.

        Returns
        -------
        list
            List of available recordings.
        """

    @abstractmethod
    def load(self, index: int):
        """Load one recording.

        Returns
        -------
        dict
        """

    @abstractmethod
    def summary(self):
        """Display dataset information."""

    # -----------------------------------------------------
    # Common Functions
    # -----------------------------------------------------
    def number_of_recordings(self):
        return len(self.recordings)

    def exists(self):
        return self.dataset_root.exists()

    def validate_file(self, file_path: Path):
        if not file_path.exists():
            raise FileNotFoundError(file_path)

    def load_edf(self, edf_path: Path):
        self.validate_file(edf_path)
        raw = mne.io.read_raw_edf(
            edf_path,
            preload=True,
            verbose=False,
        )
        return raw

    def print_recordings(self):
        print()
        print("=" * 70)
        print("Available Recordings")
        print("=" * 70)
        for i, rec in enumerate(self.recordings):
            print(f"{i + 1:03d}. {rec['name']}")
        print("=" * 70)
