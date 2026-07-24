"""
=========================================================
SleepTransitionFramework

Base Model

Parent class for all neural network models.

Every model should inherit from this class.
=========================================================
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.utils import plot_model

from src.training.metrics import MetricFactory


class BaseModel(ABC):

    """
    Abstract base class for all deep learning models.
    """

    def __init__(self):

        self.model = None

    # ---------------------------------------------------------
    # Must be implemented
    # ---------------------------------------------------------

    @abstractmethod
    def build(self):
        """
        Build the TensorFlow model.
        """
        pass

    # ---------------------------------------------------------
    # Model Access
    # ---------------------------------------------------------

    def get_model(self):

        if self.model is None:

            self.build()

        return self.model

    # ---------------------------------------------------------
    # Compile
    # ---------------------------------------------------------

    def compile(

        self,

        learning_rate=1e-3,

        optimizer=None,

        loss="binary_crossentropy",

        metrics=None

    ):

        if self.model is None:

            self.build()

        if optimizer is None:

            optimizer = tf.keras.optimizers.Adam(

                learning_rate=learning_rate

            )

        if metrics is None:

            metrics = MetricFactory.build()

        self.model.compile(

            optimizer=optimizer,

            loss=loss,

            metrics=metrics

        )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    def summary(self):

        self.get_model()

        self.model.summary()

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    def save(

        self,

        filepath="outputs/model.keras"

    ):

        self.get_model()

        filepath = Path(filepath)

        filepath.parent.mkdir(

            parents=True,

            exist_ok=True

        )

        self.model.save(filepath)

        print()

        print(f"Model saved to:\n{filepath}")

    # ---------------------------------------------------------
    # Load
    # ---------------------------------------------------------

    def load(

        self,

        filepath

    ):

        self.model = tf.keras.models.load_model(

            filepath

        )

        print()

        print(f"Loaded model:\n{filepath}")

        return self.model

    # ---------------------------------------------------------
    # Parameter Count
    # ---------------------------------------------------------

    def count_parameters(self):

        self.get_model()

        trainable = int(

            sum(

                tf.size(v)

                for v in self.model.trainable_variables

            )

        )

        non_trainable = int(

            sum(

                tf.size(v)

                for v in self.model.non_trainable_variables

            )

        )

        total = trainable + non_trainable

        return {

            "trainable": trainable,

            "non_trainable": non_trainable,

            "total": total

        }

    # ---------------------------------------------------------
    # Freeze
    # ---------------------------------------------------------

    def freeze_layers(self):

        self.get_model()

        for layer in self.model.layers:

            layer.trainable = False

    # ---------------------------------------------------------
    # Unfreeze
    # ---------------------------------------------------------

    def unfreeze_layers(self):

        self.get_model()

        for layer in self.model.layers:

            layer.trainable = True

    # ---------------------------------------------------------
    # Plot Model
    # ---------------------------------------------------------

    def plot(

        self,

        filename="outputs/model.png"

    ):

        self.get_model()

        filename = Path(filename)

        filename.parent.mkdir(

            parents=True,

            exist_ok=True

        )

        plot_model(

            self.model,

            to_file=str(filename),

            show_shapes=True,

            show_layer_names=True,

            dpi=150

        )

        print()

        print(f"Architecture saved:\n{filename}")

    # ---------------------------------------------------------
    # Export Summary
    # ---------------------------------------------------------

    def export_summary(

        self,

        filename="outputs/model_summary.txt"

    ):

        self.get_model()

        filename = Path(filename)

        filename.parent.mkdir(

            parents=True,

            exist_ok=True

        )

        with open(filename, "w") as f:

            self.model.summary(

                print_fn=lambda x: f.write(x + "\n")

            )

        print()

        print(f"Summary exported:\n{filename}")

    # ---------------------------------------------------------
    # Information
    # ---------------------------------------------------------

    def info(self):

        params = self.count_parameters()

        print()

        print("=" * 70)

        print("MODEL INFORMATION")

        print("=" * 70)

        print()

        print("Model Name")

        print(self.model.name)

        print()

        print("Trainable")

        print(params["trainable"])

        print()

        print("Non Trainable")

        print(params["non_trainable"])

        print()

        print("Total")

        print(params["total"])

        print()

        print("=" * 70)
