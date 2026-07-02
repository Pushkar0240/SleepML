"""ISRUC dataset support placeholder."""

from __future__ import annotations

from pathlib import Path

from .base_dataset import BaseDataset


class ISRUCDataset(BaseDataset):
    def discover(self):
        raise NotImplementedError("ISRUC support will be added in a later phase.")


class ISRUCRecording:
    """Placeholder recording object for the ISRUC dataset."""

    def __init__(self, *_args, **_kwargs) -> None:
        raise NotImplementedError("ISRUC recording support will be added later.")
