"""
Transition Object

Represents one detected sleep transition.
"""

from dataclasses import dataclass


@dataclass
class Transition:

    transition_type: str

    previous_stage: str

    next_stage: str

    center: int

    start: int

    end: int

    def __str__(self):

        return (
            f"{self.transition_type:25}"
            f" | Center={self.center:6d}"
            f" | Window={self.start:6d}-{self.end:6d}"
        )
