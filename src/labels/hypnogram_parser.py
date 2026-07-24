"""
SleepTransitionCNN
==================

Hypnogram Parser

Reads sleep-stage annotations from an MNE Raw object
and converts them into SleepStage objects.


"""

from __future__ import annotations

from typing import List

from src.data import Recording
from src.labels.sleep_stage import SleepStage
from src.logger import logger_factory

logger = logger_factory.get_logger("HypnogramParser")


class HypnogramParser:
    """
    Parse sleep stage annotations.
    """

    def __init__(self):
        self.stages: List[SleepStage] = []

    # ----------------------------------------------------------
    # Parse annotations
    # ----------------------------------------------------------

    def parse(self, record: Recording):
        self.stages.clear()

        if record.annotations is None:
            raise ValueError("Recording has no annotations.")

        logger.info("Reading hypnogram...")

        for onset, duration, description in zip(
            record.annotations.onset,
            record.annotations.duration,
            record.annotations.description,
        ):
            stage = SleepStage(
                stage=description,
                onset=float(onset),
                duration=float(duration),
                end=float(onset + duration),
            )
            self.stages.append(stage)

        logger.info(
            "Parsed %d sleep stages.",
            len(self.stages),
        )

        return self.stages

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------

    def summary(self):
        print()
        print("=" * 70)
        print("Sleep Stage Summary")
        print("=" * 70)

        for stage in self.stages:
            print(stage)

        print("=" * 70)
