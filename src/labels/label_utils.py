"""
Utility functions for label generation.
"""

from __future__ import annotations


def is_transition_second(second: int, transitions) -> bool:
    """
    Returns True if 'second' lies inside any transition window.
    """

    for transition in transitions:

        if transition.start <= second <= transition.end:

            return True

    return False


def transition_type(second: int, transitions):

    for transition in transitions:

        if transition.start <= second <= transition.end:

            return transition.transition_type

    return None
