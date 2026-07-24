"""
SleepTransitionCNN
==================

Channel Selector

Automatically identifies EEG, EOG, EMG, ECG and miscellaneous channels
from an MNE Raw object and allows selecting channels for further
processing.


"""

from __future__ import annotations

from typing import Dict, List

import mne

from src.data import Recording
from src.logger import logger_factory

logger = logger_factory.get_logger("ChannelSelector")


class ChannelSelector:
    """Automatically detect physiological channels."""

    def __init__(self):
        self.groups = {
            "EEG": [],
            "EOG": [],
            "EMG": [],
            "ECG": [],
            "MISC": [],
        }

    # -------------------------------------------------------
    # Detect channel groups
    # -------------------------------------------------------

    def detect(self, record: Recording) -> Dict[str, List[str]]:
        if record.raw is None:
            raise ValueError("Recording has no loaded EDF.")

        # Clear previous groups.
        for key in self.groups:
            self.groups[key] = []

        for channel in record.raw.ch_names:
            name = channel.lower()

            if "eeg" in name:
                self.groups["EEG"].append(channel)
            elif "eog" in name:
                self.groups["EOG"].append(channel)
            elif "emg" in name:
                self.groups["EMG"].append(channel)
            elif "ecg" in name or "ekg" in name:
                self.groups["ECG"].append(channel)
            else:
                self.groups["MISC"].append(channel)

        # Update Recording object.
        record.eeg_channels = self.groups["EEG"]
        record.eog_channels = self.groups["EOG"]
        record.emg_channels = self.groups["EMG"]
        record.ecg_channels = self.groups["ECG"]
        record.misc_channels = self.groups["MISC"]
        record.processing_history.append("Detected channel groups")

        logger.info("Channel detection completed.")
        logger.info(
            "EEG=%d EOG=%d EMG=%d ECG=%d MISC=%d",
            len(record.eeg_channels),
            len(record.eog_channels),
            len(record.emg_channels),
            len(record.ecg_channels),
            len(record.misc_channels),
        )

        return self.groups

    # -------------------------------------------------------
    # Print groups
    # -------------------------------------------------------

    def summary(self):
        print("\nDetected Channel Groups")
        print("=" * 60)

        for group, channels in self.groups.items():
            print(f"\n{group} ({len(channels)})")

            for ch in channels:
                print(f"   {ch}")

    # -------------------------------------------------------
    # Select channels
    # -------------------------------------------------------

    def get_selected_channels(
        self,
        eeg=True,
        eog=False,
        emg=False,
        ecg=False,
    ) -> List[str]:
        selected = []

        if eeg:
            selected.extend(self.groups["EEG"])

        if eog:
            selected.extend(self.groups["EOG"])

        if emg:
            selected.extend(self.groups["EMG"])

        if ecg:
            selected.extend(self.groups["ECG"])

        return selected

    # -------------------------------------------------------
    # Return filtered Raw object
    # -------------------------------------------------------

    def pick_channels(
        self,
        record: Recording,
        eeg=True,
        eog=False,
        emg=False,
        ecg=False,
    ) -> mne.io.BaseRaw:
        selected = self.get_selected_channels(
            eeg=eeg,
            eog=eog,
            emg=emg,
            ecg=ecg,
        )

        logger.info("Selected channels: %s", selected)

        if selected:
            record.processing_history.append(f"Selected channels: {', '.join(selected)}")

        return record.raw.copy().pick(selected)
