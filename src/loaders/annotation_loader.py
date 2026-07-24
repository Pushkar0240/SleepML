"""
=========================================================
SleepTransitionFramework

Annotation Loader

Loads and parses ANPHY annotation files.
=========================================================
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd


class AnnotationLoader:
    """
    Loads and parses ANPHY annotation files.
    """
    
    # Define valid sleep stages based on typical ANPHY/AASM conventions
    # W = Wake, N1/N2/N3 = NREM, R = REM, L = Lights on, M = Movement, ? = Unknown
    VALID_STAGES = ["W", "N1", "N2", "N3", "R", "L", "M", "?"]

    def __init__(self):
        self.annotations = None

    def load(
        self,
        filename
    ):
        """
        Load ANPHY annotation file.

        File format
        -----------

        Stage   Start   Duration

        W       0       30
        W       30      30
        N1      60      30
        ...
        """

        filename = Path(filename)

        if not filename.exists():

            raise FileNotFoundError(filename)

        df = pd.read_csv(

            filename,

            sep=r"\s+",

            names=[

                "Stage",

                "Start",

                "Duration"

            ],

            header=None

        )

        df["Start"] = df["Start"].astype(int)

        df["Duration"] = df["Duration"].astype(int)

        self.annotations = df

        self._validate()

        return df

    # -----------------------------------------------------

    def _validate(self):

        if self.annotations is None:

            raise ValueError(

                "No annotation loaded."

            )

        invalid = self.annotations[

            ~self.annotations["Stage"].isin(

                self.VALID_STAGES

            )

        ]

        if len(invalid):

            raise ValueError(

                f"Unknown sleep stages found:\n{invalid}"

            )

        if (self.annotations["Duration"] <= 0).any():

            raise ValueError(

                "Invalid epoch duration."

            )

    # -----------------------------------------------------

    def start_times(self):

        return self.annotations["Start"].to_numpy()

    # -----------------------------------------------------

    def durations(self):

        return self.annotations["Duration"].to_numpy()

    # -----------------------------------------------------

    def stages_numpy(self):

        return self.annotations["Stage"].to_numpy()

    # -----------------------------------------------------

    def stage_at_time(

        self,

        seconds

    ):

        row = self.annotations[

            self.annotations["Start"] == seconds

        ]

        if len(row) == 0:

            return None

        return row.iloc[0]["Stage"]

    # -----------------------------------------------------

    def total_duration(self):

        last = self.annotations.iloc[-1]

        return (

            last["Start"]

            +

            last["Duration"]

        )
