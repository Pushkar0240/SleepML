"""
SleepTransitionCNN
==================

Normalization Module

Supported Methods
-----------------
1. Z-score
2. Min-Max
3. Robust Scaling

Author:
Pushkar Singh
"""

from __future__ import annotations

import numpy as np

from src.config import config
from src.data import Recording
from src.logger import logger_factory

logger = logger_factory.get_logger("Normalize")


class Normalizer:
    """Normalize EEG signals."""

    def __init__(self):
        self.method = config.get("preprocessing", "normalization", "method").lower()

    # --------------------------------------------------------
    # Z-score
    # --------------------------------------------------------

    def zscore(self, signal):
        mean = np.mean(signal, axis=1, keepdims=True)
        std = np.std(signal, axis=1, keepdims=True)
        std[std == 0] = 1
        return (signal - mean) / std

    # --------------------------------------------------------
    # Min-Max
    # --------------------------------------------------------

    def minmax(self, signal):
        minimum = np.min(signal, axis=1, keepdims=True)
        maximum = np.max(signal, axis=1, keepdims=True)
        denominator = maximum - minimum
        denominator[denominator == 0] = 1
        return (signal - minimum) / denominator

    # --------------------------------------------------------
    # Robust
    # --------------------------------------------------------

    def robust(self, signal):
        median = np.median(signal, axis=1, keepdims=True)
        q75 = np.percentile(signal, 75, axis=1, keepdims=True)
        q25 = np.percentile(signal, 25, axis=1, keepdims=True)
        iqr = q75 - q25
        iqr[iqr == 0] = 1
        return (signal - median) / iqr

    # --------------------------------------------------------
    # Apply
    # --------------------------------------------------------

    def apply(self, record: Recording) -> Recording:
        logger.info("Normalization method: %s", self.method)

        signal = record.signal
        if signal is None:
            raise ValueError("Signal has not been loaded.")

        if self.method == "zscore":
            signal = self.zscore(signal)
        elif self.method == "minmax":
            signal = self.minmax(signal)
        elif self.method == "robust":
            signal = self.robust(signal)
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")

        record.signal = signal
        record.processing_history.append(f"Normalized ({self.method})")

        logger.info("Normalization completed.")

        return record
