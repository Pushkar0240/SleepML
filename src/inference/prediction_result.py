"""
=========================================================
SleepTransitionFramework

Prediction Result

Represents the output of a model prediction.
=========================================================
"""

from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class PredictionResult:
    """
    Represents a single prediction.
    """

    probability: float

    predicted_label: int

    confidence: float

    threshold: float

    model_name: str

    sample_id: str | None = None

    subject_id: str | None = None

    recording_id: str | None = None

    ground_truth: int | None = None

    # ---------------------------------------------------------

    @property
    def prediction(self):

        return (

            "Transition"

            if self.predicted_label == 1

            else

            "Non-Transition"

        )

    # ---------------------------------------------------------

    @property
    def correct(self):

        if self.ground_truth is None:

            return None

        return self.predicted_label == self.ground_truth

    # ---------------------------------------------------------

    def to_dict(self):

        data = asdict(self)

        data["prediction"] = self.prediction

        data["correct"] = self.correct

        return data

    # ---------------------------------------------------------

    def summary(self):

        print()

        print("=" * 70)

        print("PREDICTION RESULT")

        print("=" * 70)

        print()

        print("Prediction")

        print(self.prediction)

        print()

        print("Probability")

        print(f"{self.probability:.4f}")

        print()

        print("Confidence")

        print(f"{self.confidence:.4f}")

        print()

        if self.ground_truth is not None:

            print("Ground Truth")

            print(self.ground_truth)

            print()

            print("Correct")

            print(self.correct)

            print()

        print("=" * 70)
