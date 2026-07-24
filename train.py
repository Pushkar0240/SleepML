"""
=========================================================
SleepTransitionFramework

Training Script

Entry point for training.
=========================================================
"""

import tensorflow as tf

from src.config import ConfigManager
from src.models import ModelFactory
from src.training.trainer import ModelTrainer


def print_environment():

    print()

    print("=" * 70)

    print("TRAINING ENVIRONMENT")

    print("=" * 70)

    print()

    print("TensorFlow :", tf.__version__)

    gpus = tf.config.list_physical_devices("GPU")

    print()

    if gpus:

        print("GPU Available")

        for gpu in gpus:

            print("-", gpu.name)

    else:

        print("Running on CPU")

    print()


def main():

    config = ConfigManager()

    print_environment()

    config.summary()

    model = ModelFactory.create(

        config.model["name"],
        
        config=config

    )

    trainer = ModelTrainer(

        model=model.get_model(),

        config=config

    )

    trainer.train()

    trainer.save()


if __name__ == "__main__":

    main()
