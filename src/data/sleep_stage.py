"""Sleep stage labels and transition helpers."""

from __future__ import annotations

from enum import Enum


class SleepStage(str, Enum):
    """Canonical sleep stage labels."""

    W = "W"
    N1 = "N1"
    N2 = "N2"
    N3 = "N3"
    R = "R"
    UNKNOWN = "UNKNOWN"
