"""
SleepTransitionCNN
==================

Automatic Transition Detector

Detects every sleep stage change from the
per-second timeline.


"""

from __future__ import annotations

from typing import List

from src.labels.transition import Transition
from src.logger import logger_factory

logger = logger_factory.get_logger("TransitionDetector")


class TransitionDetector:

    """
    Detect sleep stage transitions.
    """

    def __init__(self, window_size: int = 30):

        self.window_size = window_size

        self.transitions: List[Transition] = []

    # ---------------------------------------------------------
    # Detect transitions
    # ---------------------------------------------------------

    def detect(self, timeline):

        self.transitions.clear()

        logger.info("Searching for transitions...")

        for second in range(1, len(timeline)):

            previous_stage = timeline[second - 1]

            current_stage = timeline[second]

            if previous_stage == current_stage:
                continue

            start = max(
                0,
                second - self.window_size
            )

            end = min(
                len(timeline),
                second + self.window_size
            )

            transition = Transition(

                transition_type=f"{previous_stage} → {current_stage}",

                previous_stage=previous_stage,

                next_stage=current_stage,

                center=second,

                start=start,

                end=end

            )

            self.transitions.append(
                transition
            )

        logger.info(

            "%d transitions detected.",

            len(self.transitions)

        )

        return self.transitions

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    def summary(self):

        print()

        print("=" * 80)

        print("Detected Transitions")

        print("=" * 80)

        for transition in self.transitions:

            print(transition)

        print("=" * 80)
