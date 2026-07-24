"""
SleepTransitionCNN
==================

Preprocessing Pipeline

Applies all preprocessing steps to a Recording.

Pipeline

Raw EDF
    ↓
Channel Selection
    ↓
Resampling
    ↓
Bandpass Filter
    ↓
Notch Filter
    ↓
Normalization
    ↓
Processed Recording
"""

from __future__ import annotations

import mne

from src.data.recording import Recording


class PreprocessingPipeline:

    def __init__(
        self,
        config: dict = None,
        target_sampling_rate: int = 100,
        low_cutoff: float = 0.3,
        high_cutoff: float = 35.0,
        notch_frequency: float = 50.0,
        normalize: bool = False
    ):
        if config is not None and "preprocessing" in config:
            prep_cfg = config["preprocessing"]
            self.target_sampling_rate = prep_cfg.get("target_sampling_rate", target_sampling_rate)
            
            bandpass = prep_cfg.get("bandpass", {})
            self.low_cutoff = bandpass.get("low", low_cutoff)
            self.high_cutoff = bandpass.get("high", high_cutoff)
            
            self.notch_frequency = prep_cfg.get("notch", notch_frequency)
            self.normalize = prep_cfg.get("normalize", normalize)
        else:
            self.target_sampling_rate = target_sampling_rate
            self.low_cutoff = low_cutoff
            self.high_cutoff = high_cutoff
            self.notch_frequency = notch_frequency
            self.normalize = normalize

    # -----------------------------------------------------

    def process(self, record: Recording) -> Recording:

        raw = record.raw

        if raw is None:

            raise RuntimeError(
                "Recording.raw is empty."
            )

        print("\nPreprocessing Recording...")

        # ------------------------------------------
        # Resample
        # ------------------------------------------

        if raw.info["sfreq"] != self.target_sampling_rate:

            raw.resample(

                self.target_sampling_rate

            )

            record.add_history(

                f"Resampled ({self.target_sampling_rate} Hz)"

            )

        # ------------------------------------------
        # Bandpass Filter
        # ------------------------------------------

        raw.filter(

            l_freq=self.low_cutoff,

            h_freq=self.high_cutoff,

            verbose=False

        )

        record.add_history(

            f"Bandpass {self.low_cutoff}-{self.high_cutoff} Hz"

        )

        # ------------------------------------------
        # Notch Filter
        # ------------------------------------------

        nyquist = raw.info["sfreq"] / 2

        if self.notch_frequency < nyquist:

            raw.notch_filter(

                self.notch_frequency,

                verbose=False

            )

            record.add_history(

                f"Notch {self.notch_frequency} Hz"

            )

        # ------------------------------------------
        # Update Recording
        # ------------------------------------------

        record.raw = raw

        record.update_from_raw()

        # ------------------------------------------
        # Optional Normalization
        # ------------------------------------------

        if self.normalize:

            signal = record.signal

            mean = signal.mean(

                axis=1,

                keepdims=True

            )

            std = signal.std(

                axis=1,

                keepdims=True

            )

            std[std == 0] = 1

            record.signal = (

                signal - mean

            ) / std

            record.add_history(

                "Z-score Normalization"

            )

        print("Preprocessing Complete")

        return record
