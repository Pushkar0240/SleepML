"""
SleepTransitionCNN
==================

Stage Timeline Generator

Converts SleepStage annotations into
a per-second timeline.
"""

from __future__ import annotations

from collections import Counter

from src.labels.sleep_stage import SleepStage
from src.logger import logger_factory

logger = logger_factory.get_logger("StageTimeline")


class StageTimeline:

    def __init__(self):

        self.timeline = []

    # ----------------------------------------------------
    # Build timeline
    # ----------------------------------------------------

    def build(self, stages: list[SleepStage]):

        self.timeline.clear()

        logger.info("Building per-second timeline...")

        for stage in stages:

            start = int(stage.onset)

            end = int(stage.end)

            for _ in range(start, end):

                self.timeline.append(stage.stage)

        logger.info(
            "Timeline length : %d seconds",
            len(self.timeline)
        )

        return self.timeline

    # ----------------------------------------------------
    # Statistics
    # ----------------------------------------------------

    def statistics(self):

        print()

        print("=" * 60)

        print("Timeline Statistics")

        print("=" * 60)

        counter = Counter(self.timeline)

        for stage, count in counter.items():

            print(f"{stage:20} {count:8d} sec")

        print("=" * 60)

    # ----------------------------------------------------
    # Preview
    # ----------------------------------------------------

    def preview(self, seconds=30):

        print()

        print("=" * 60)

        print("Timeline Preview")

        print("=" * 60)

        for second, stage in enumerate(
                self.timeline[:seconds]):

            print(f"{second:5d}  {stage}")
