"""
SleepTransitionFramework
========================

Model Trainer

Generic training engine for all TensorFlow models.


"""

from __future__ import annotations

from pathlib import Path

import tensorflow as tf

from src.training.dataloader import EEGDataLoader
from src.training.callbacks import CallbackFactory
from src.training.metrics import MetricFactory
from src.training.history import HistoryManager


class ModelTrainer:

    def __init__(

        self,

        model,

        config,

        experiment=None

    ):

        self.model = model

        self.config = config
        
        self.experiment = experiment

        self.dataset_root = Path(config.dataset["root"])

        self.batch_size = config.training["batch_size"]

        self.epochs = config.training["epochs"]

        self.learning_rate = config.training["learning_rate"]

    # -------------------------------------------------

    def compile(self):

        self.model.compile(

            optimizer=tf.keras.optimizers.Adam(

                learning_rate=self.learning_rate

            ),

            loss=tf.keras.losses.BinaryCrossentropy(),

            metrics=MetricFactory.build()

        )

    # -------------------------------------------------

    def load_data(self):

        train_loader = EEGDataLoader(

            self.dataset_root / "train",

            batch_size=self.batch_size,

            shuffle=True

        )

        validation_loader = EEGDataLoader(

            self.dataset_root / "validation",

            batch_size=self.batch_size,

            shuffle=False

        )

        self.train_dataset = train_loader.dataset()

        self.validation_dataset = validation_loader.dataset()

    # -------------------------------------------------

    def train(self):

        self.compile()

        self.load_data()

        callbacks = CallbackFactory(self.config).build()

        print()

        print("=" * 70)

        print("Training Started")

        print("=" * 70)

        print()

        history = self.model.fit(

            self.train_dataset,

            validation_data=self.validation_dataset,

            epochs=self.epochs,

            callbacks=callbacks,

            verbose=1

        )

        HistoryManager().save(

            history

        )

        print()

        print("=" * 70)

        print("Training Finished")

        print("=" * 70)

        print()

        return history

    # -------------------------------------------------

    def summary(self):

        self.model.summary()

    # -------------------------------------------------

    def save(

        self,

        filename="outputs/final_model.keras"

    ):

        self.model.save(

            filename

        )

        print()

        print(

            f"Model saved -> {filename}"

        )
