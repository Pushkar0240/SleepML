"""
SleepTransitionCNN
==================

Dataset Builder

Creates a unified list of TrainingSample objects.
"""

from __future__ import annotations

import random
from typing import List

from src.data.training_sample import TrainingSample
from src.labels.window_generator import WindowGenerator


def overlaps_transition(second: int, transitions, margin: int = 30):
    for t in transitions:
        if abs(second - int(t.center)) <= margin:
            return True
    return False


class DatasetBuilder:

    def __init__(self, window_size: int = 30):

        self.window_size = window_size

        self.generator = WindowGenerator(
            window_size
        )

    # ---------------------------------------------------------

    def build(
        self,
        record,
        transitions,
        negative_ratio: float = 1.0
    ) -> List[TrainingSample]:

        samples = self.generator.generate(
            record,
            transitions
        )

        fs = int(record.sampling_rate)

        signal = record.signal

        duration = int(record.duration)

        desired_negative = int(
            len(samples) * negative_ratio
        )
        
        num_transitions = len(transitions)
        
        while len(samples) < (desired_negative + num_transitions):

            second = random.randint(

                self.window_size,

                duration - self.window_size

            )

            if overlaps_transition(second, transitions, self.window_size):
                continue

            start = second - self.window_size

            end = second + self.window_size

            start_sample = start * fs

            end_sample = end * fs

            if end_sample > signal.shape[1]:

                continue

            window = signal[
                :,
                start_sample:end_sample
            ]

            samples.append(

                TrainingSample(

                    signal=window,

                    label=0,

                    subject_id=record.subject_id,

                    recording_id=record.recording_id,

                    dataset_name=record.dataset_name,

                    transition_type="Stable",

                    previous_stage="",

                    next_stage="",

                    start_second=start,

                    center_second=second,

                    end_second=end

                )

            )

        record.training_samples = samples

        return samples
