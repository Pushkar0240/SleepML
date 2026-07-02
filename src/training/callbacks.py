"""Training callbacks."""

from __future__ import annotations


class TrainingCallback:
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("Training callbacks will be added in a later phase.")
