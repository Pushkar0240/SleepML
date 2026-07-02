"""
SleepTransitionCNN
==================

Signal Filtering Module

Provides reusable filtering operations for EEG recordings.

Features
--------
- Band-pass filtering
- Low-pass filtering
- High-pass filtering
- Notch filtering
- Automatic configuration from config.yaml

Author:
Pushkar Singh
"""

from __future__ import annotations

from src.config import config
from src.data import Recording
from src.logger import logger_factory

logger = logger_factory.get_logger("Filters")


class SignalFilter:
    """Signal preprocessing filters."""

    def __init__(self):
        self.low = config.get("preprocessing", "bandpass", "low")
        self.high = config.get("preprocessing", "bandpass", "high")
        self.notch = config.get("preprocessing", "notch", "frequency")

    def _update_signal(self, record: Recording) -> Recording:
        """Refresh cached signal data after an in-place Raw operation."""

        record.signal = record.raw.get_data() if record.raw is not None else None
        return record

    def _add_history(self, record: Recording, entry: str) -> None:
        record.processing_history.append(entry)

    # ---------------------------------------------------------
    # Band-pass
    # ---------------------------------------------------------

    def bandpass(self, record: Recording) -> Recording:
        if record.raw is None:
            raise ValueError("Recording has no loaded EDF.")

        logger.info("Band-pass filtering %.2f - %.2f Hz", self.low, self.high)

        record.raw.filter(l_freq=self.low, h_freq=self.high, verbose=False)
        self._update_signal(record)
        self._add_history(record, f"Band-pass: {self.low:.2f}-{self.high:.2f} Hz")

        return record

    # ---------------------------------------------------------
    # High-pass
    # ---------------------------------------------------------

    def highpass(self, record: Recording, cutoff: float) -> Recording:
        if record.raw is None:
            raise ValueError("Recording has no loaded EDF.")

        logger.info("High-pass %.2f Hz", cutoff)

        record.raw.filter(l_freq=cutoff, h_freq=None, verbose=False)
        self._update_signal(record)
        self._add_history(record, f"High-pass: {cutoff:.2f} Hz")

        return record

    # ---------------------------------------------------------
    # Low-pass
    # ---------------------------------------------------------

    def lowpass(self, record: Recording, cutoff: float) -> Recording:
        if record.raw is None:
            raise ValueError("Recording has no loaded EDF.")

        logger.info("Low-pass %.2f Hz", cutoff)

        record.raw.filter(l_freq=None, h_freq=cutoff, verbose=False)
        self._update_signal(record)
        self._add_history(record, f"Low-pass: {cutoff:.2f} Hz")

        return record

    # ---------------------------------------------------------
    # Notch
    # ---------------------------------------------------------

    def notch_filter(self, record: Recording) -> Recording:
        if record.raw is None:
            raise ValueError("Recording has no loaded EDF.")

        nyquist = float(record.raw.info["sfreq"]) / 2.0
        logger.info("Applying %.1f Hz notch filter", self.notch)

        if self.notch >= nyquist:
            logger.warning(
                "Skipping notch filter at %.1f Hz because Nyquist is %.1f Hz",
                self.notch,
                nyquist,
            )
            self._add_history(record, f"Notch skipped: {self.notch:.1f} Hz >= Nyquist")
            return record

        record.raw.notch_filter(freqs=self.notch, verbose=False)
        self._update_signal(record)
        self._add_history(record, f"Notch: {self.notch:.1f} Hz")

        return record

    # ---------------------------------------------------------
    # Complete filtering pipeline
    # ---------------------------------------------------------

    def apply(self, record: Recording) -> Recording:
        if record.raw is None:
            raise ValueError("Recording has no loaded EDF.")

        logger.info("Starting filtering pipeline")

        record = self.bandpass(record)
        record = self.notch_filter(record)

        logger.info("Filtering completed")

        return record
