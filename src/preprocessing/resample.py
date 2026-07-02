"""
SleepTransitionCNN
==================

Resampling Module

Resamples EEG recordings to a common sampling frequency.

Author:
Pushkar Singh
"""

from __future__ import annotations

from src.config import config
from src.data import Recording
from src.logger import logger_factory

logger = logger_factory.get_logger("Resample")


class Resampler:
    """Resample recordings to the configured sampling rate."""

    def __init__(self):
        self.target_rate = config.get("preprocessing", "sampling_rate")

    # ---------------------------------------------------------
    # Resample Recording
    # ---------------------------------------------------------

    def apply(self, record: Recording) -> Recording:
        if record.raw is None:
            raise ValueError("Recording has no loaded Raw object.")

        current_rate = float(record.raw.info["sfreq"])

        logger.info("Current Sampling Rate : %.2f Hz", current_rate)

        if current_rate == float(self.target_rate):
            logger.info("Recording already sampled at %.2f Hz", self.target_rate)
            record.sampling_rate = current_rate
            record.signal = record.raw.get_data()
            record.processing_history.append(f"Resampled to {self.target_rate} Hz")
            return record

        logger.info(
            "Resampling %.2f Hz → %.2f Hz",
            current_rate,
            self.target_rate,
        )

        record.raw.resample(sfreq=self.target_rate, verbose=False)
        record.signal = record.raw.get_data()
        record.sampling_rate = float(record.raw.info["sfreq"])
        record.processing_history.append(f"Resampled to {self.target_rate} Hz")

        logger.info("Resampling completed.")

        return record
