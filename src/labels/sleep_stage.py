"""
SleepTransitionCNN
==================

Sleep Stage Object

Represents one sleep-stage annotation read from
a hypnogram EDF file.
"""

from dataclasses import dataclass


@dataclass
class SleepStage:
    """
    Represents a single sleep stage annotation.
    """

    stage: str
    onset: float
    duration: float
    end: float

    def __str__(self):
        return (
            f"{self.stage:12}"
            f" | {self.onset:8.2f}"
            f" -> {self.end:8.2f}"
            f" ({self.duration:6.2f}s)"
        )
