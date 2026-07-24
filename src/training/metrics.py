"""
SleepTransitionFramework
========================

Training Metrics

Defines TensorFlow metrics used while training.


"""

from __future__ import annotations

import tensorflow as tf


class MetricFactory:
    """
    Factory class for TensorFlow metrics.
    """

    @staticmethod
    def accuracy():

        return tf.keras.metrics.BinaryAccuracy(
            name="accuracy"
        )

    @staticmethod
    def precision():

        return tf.keras.metrics.Precision(
            name="precision"
        )

    @staticmethod
    def recall():

        return tf.keras.metrics.Recall(
            name="recall"
        )

    @staticmethod
    def auc():

        return tf.keras.metrics.AUC(
            name="auc"
        )

    @classmethod
    def build(cls):
        """
        Return all metrics used during model.fit().
        """

        return [

            cls.accuracy(),

            cls.precision(),

            cls.recall(),

            cls.auc()

        ]


def metric_names():

    return [

        "loss",

        "accuracy",

        "precision",

        "recall",

        "auc"

    ]
