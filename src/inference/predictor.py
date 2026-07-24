"""
=========================================================
SleepTransitionFramework

Predictor

Loads a trained model and performs inference.

Responsibilities
----------------
- Load trained model
- Predict single sample
- Predict batch
- Return PredictionResult objects
=========================================================
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf

from src.inference.prediction_result import PredictionResult


class Predictor:
    """
    Generic inference engine.
    """

    # ---------------------------------------------------------

    def __init__(

        self,

        model_path="outputs/models/best_model.keras",

        threshold=0.50

    ):

        self.model_path = Path(model_path)

        self.threshold = threshold

        self.model = None

        self.load_model()

    # ---------------------------------------------------------

    def load_model(self):

        """
        Load trained TensorFlow model.
        """

        if not self.model_path.exists():

            raise FileNotFoundError(

                f"Model not found:\n{self.model_path}"

            )

        self.model = tf.keras.models.load_model(

            self.model_path

        )

    # ---------------------------------------------------------

    def predict(

        self,

        sample,

        sample_id=None,

        subject_id=None,

        recording_id=None,

        ground_truth=None

    ):

        """
        Predict a single EEG window.
        """

        sample = np.asarray(sample)

        if sample.ndim != 2:

            raise ValueError(

                "Expected input shape (time, channels)."

            )

        sample = np.expand_dims(

            sample,

            axis=0

        )

        probability = float(

            self.model.predict(

                sample,

                verbose=0

            )[0][0]

        )

        prediction = int(

            probability >= self.threshold

        )

        confidence = (

            probability

            if prediction == 1

            else

            1.0 - probability

        )

        return PredictionResult(

            probability=probability,

            predicted_label=prediction,

            confidence=confidence,

            threshold=self.threshold,

            model_name=self.model.name,

            sample_id=sample_id,

            subject_id=subject_id,

            recording_id=recording_id,

            ground_truth=ground_truth

        )

    # ---------------------------------------------------------

    def predict_batch(

        self,

        samples

    ):

        """
        Predict multiple EEG windows.
        """

        samples = np.asarray(samples)

        if samples.ndim != 3:

            raise ValueError(

                "Expected shape (N, time, channels)."

            )

        if samples.shape[1] < samples.shape[2]:
            samples = np.transpose(samples, (0, 2, 1))
            
        probabilities = self.model.predict(

            samples,

            verbose=0

        )

        results = []

        for probability in probabilities:

            p = float(probability[0])

            label = int(

                p >= self.threshold

            )

            confidence = (

                p

                if label == 1

                else

                1.0 - p

            )

            results.append(

                PredictionResult(

                    probability=p,

                    predicted_label=label,

                    confidence=confidence,

                    threshold=self.threshold,

                    model_name=self.model.name

                )

            )

        return results

    # ---------------------------------------------------------

    def predict_probability(

        self,

        sample

    ):

        """
        Return only probability.
        """

        sample = np.asarray(sample)

        sample = np.expand_dims(

            sample,

            axis=0

        )

        return float(

            self.model.predict(

                sample,

                verbose=0

            )[0][0]

        )

    # ---------------------------------------------------------

    def predict_label(

        self,

        sample

    ):

        """
        Return only predicted label.
        """

        probability = self.predict_probability(

            sample

        )

        return int(

            probability >= self.threshold

        )

    # ---------------------------------------------------------

    def model_summary(self):

        self.model.summary()

    # ---------------------------------------------------------

    @property
    def input_shape(self):

        return self.model.input_shape

    # ---------------------------------------------------------

    @property
    def output_shape(self):

        return self.model.output_shape
