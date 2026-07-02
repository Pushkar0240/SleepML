"""MASS dataset support placeholder."""

from __future__ import annotations

from pathlib import Path

from .base_dataset import BaseDataset


class MASSDataset(BaseDataset):
    def discover(self):
        raise NotImplementedError("MASS support will be added in a later phase.")


class MASSRecording:
    """Placeholder recording object for the MASS dataset."""

    def __init__(self, *_args, **_kwargs) -> None:
        raise NotImplementedError("MASS recording support will be added later.")
