"""
SleepTransitionFramework
========================

TensorFlow Callbacks

Centralized callback factory for training.


"""

from __future__ import annotations

from pathlib import Path

import tensorflow as tf


class CallbackFactory:

    def __init__(

        self,

        config,

        monitor="val_loss"

    ):

        self.config = config

        self.output_dir = Path(config.output["root"])

        self.output_dir.mkdir(

            parents=True,

            exist_ok=True

        )

        self.monitor = monitor

    # -----------------------------------------------------

    def checkpoint(self):

        return tf.keras.callbacks.ModelCheckpoint(

            filepath=str(

                self.output_dir /

                "best_model.keras"

            ),

            monitor=self.monitor,

            save_best_only=True,

            verbose=1

        )

    # -----------------------------------------------------

    def early_stopping(self):

        return tf.keras.callbacks.EarlyStopping(

            monitor=self.monitor,

            patience=self.config.callbacks["early_stop"],

            restore_best_weights=True,

            verbose=1

        )

    # -----------------------------------------------------

    def reduce_lr(self):

        return tf.keras.callbacks.ReduceLROnPlateau(

            monitor=self.monitor,

            factor=self.config.callbacks.get("reduce_lr_factor", 0.5),

            patience=self.config.callbacks["reduce_lr"],

            min_lr=1e-6,

            verbose=1

        )

    # -----------------------------------------------------

    def csv_logger(self):

        return tf.keras.callbacks.CSVLogger(

            str(

                self.output_dir /

                "training_log.csv"

            )

        )

    # -----------------------------------------------------

    def tensorboard(self):

        return tf.keras.callbacks.TensorBoard(

            log_dir=str(

                self.output_dir /

                "tensorboard"

            ),

            histogram_freq=1

        )

    # -----------------------------------------------------

    def build(self):

        """
        Return all callbacks.
        """

        return [

            self.checkpoint(),

            self.early_stopping(),

            self.reduce_lr(),

            self.csv_logger(),

            self.tensorboard()

        ]
