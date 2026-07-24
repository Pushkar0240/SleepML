"""
SleepTransitionCNN
==================

Window Generator

Generates TrainingSample objects around detected transitions.
"""

from __future__ import annotations

from typing import List

from src.data.training_sample import TrainingSample


class WindowGenerator:

    def __init__(self, window_size: int = 30):

        self.window_size = window_size

    # ---------------------------------------------------------

    def generate(
        self,
        record,
        transitions
    ) -> List[TrainingSample]:

        samples = []

        fs = int(record.sampling_rate)

        signal = record.signal

        total_samples = signal.shape[1]

        for transition in transitions:

            center = int(transition.center)

            start = center - self.window_size

            end = center + self.window_size

            if start < 0:
                continue

            start_sample = start * fs

            end_sample = end * fs

            if end_sample > total_samples:
                continue

            window = signal[
                :,
                start_sample:end_sample
            ]

            sample = TrainingSample(

                signal=window,

                label=1,

                subject_id=record.subject_id,

                recording_id=record.recording_id,

                dataset_name=record.dataset_name,

                transition_type=transition.transition_type,

                previous_stage=transition.previous_stage,

                next_stage=transition.next_stage,

                start_second=start,

                center_second=center,

                end_second=end

            )

            samples.append(sample)

        return samples
