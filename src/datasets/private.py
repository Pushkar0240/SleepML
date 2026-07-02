"""Private dataset support placeholder."""

from __future__ import annotations

from pathlib import Path

from .base_dataset import BaseDataset


class PrivateDataset(BaseDataset):
    def discover(self):
        raise NotImplementedError("Private dataset support will be defined later.")


class PrivateRecording:
    """Recording interface for user-provided EEG data."""

    def __init__(self, *_args, **_kwargs) -> None:
        raise NotImplementedError("Private recording support will be defined later.")
