"""
SleepTransitionCNN
==================

Recording Object

Represents one complete overnight sleep recording.

A Recording object is dataset-independent. Whether the data comes from
Sleep-EDF, MASS, ISRUC, or a private laboratory recording, every loader
returns the same object.

Author
------
Pushkar Singh
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mne
import numpy as np


@dataclass
class Recording:
    """
    Represents one overnight sleep recording.
    """

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    name: str
    dataset: str
    psg_path: Path
    hypnogram_path: Optional[Path] = None

    # ------------------------------------------------------------------
    # Raw data
    # ------------------------------------------------------------------

    raw: Optional[mne.io.BaseRaw] = None
    annotations: Optional[mne.Annotations] = None

    # ------------------------------------------------------------------
    # Signals
    # ------------------------------------------------------------------

    signal: Optional[np.ndarray] = None

    sampling_rate: float = 0.0

    duration: float = 0.0

    # ------------------------------------------------------------------
    # Channel Groups
    # ------------------------------------------------------------------

    eeg_channels: List[str] = field(default_factory=list)

    eog_channels: List[str] = field(default_factory=list)

    emg_channels: List[str] = field(default_factory=list)

    ecg_channels: List[str] = field(default_factory=list)

    misc_channels: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Labels
    # ------------------------------------------------------------------

    sleep_stage_labels: List[str] = field(default_factory=list)

    transition_intervals: List[Tuple[float, float, str]] = field(default_factory=list)

    transition_labels: Optional[np.ndarray] = None

    prediction_probabilities: Optional[np.ndarray] = None

    prediction_labels: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Windows
    # ------------------------------------------------------------------

    windows: Optional[np.ndarray] = None

    window_labels: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Features
    # ------------------------------------------------------------------

    features: Dict[str, Any] = field(default_factory=dict)

    metadata: Dict[str, Any] = field(default_factory=dict)

    processing_history: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------

    def is_loaded(self) -> bool:
        """Return True if EDF data has been loaded."""
        return self.raw is not None

    def number_of_channels(self) -> int:
        """Return total number of channels."""
        if self.raw is None:
            return 0
        return len(self.raw.ch_names)

    def number_of_samples(self) -> int:
        """Return total number of samples."""
        if self.signal is None:
            return 0
        return self.signal.shape[-1]

    @property
    def sleep_stages(self) -> List[str]:
        """Backward-compatible alias for sleep stage labels."""

        return self.sleep_stage_labels

    @sleep_stages.setter
    def sleep_stages(self, values: List[str]) -> None:
        self.sleep_stage_labels = list(values)

    def summary(self) -> None:
        """Print recording information."""

        print("=" * 70)
        print("Recording Summary")
        print("=" * 70)

        print(f"Name            : {self.name}")
        print(f"Dataset         : {self.dataset}")
        print(f"Sampling Rate   : {self.sampling_rate:.2f} Hz")
        print(f"Duration        : {self.duration:.2f} seconds")
        print(f"Channels        : {self.number_of_channels()}")
        print(f"Stages          : {len(self.sleep_stage_labels)}")
        print(f"Transitions     : {len(self.transition_intervals)}")

        print()
        print("Channel Groups")
        print("---------------------------")
        print(f"EEG : {len(self.eeg_channels)}")
        print(f"EOG : {len(self.eog_channels)}")
        print(f"EMG : {len(self.emg_channels)}")
        print(f"ECG : {len(self.ecg_channels)}")
        print(f"MISC: {len(self.misc_channels)}")

        if self.features:
            print()
            print("Features")
            print("---------------------------")
            print(f"Keys: {', '.join(sorted(self.features.keys()))}")

        print("=" * 70)
