"""
=========================================================
SleepTransitionFramework

Prediction Script

Entry point for inference on processed EEG data.
=========================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ConfigManager
from src.inference.predictor import Predictor
from src.inference.inference_pipeline import InferencePipeline


def load_dataset(config):
    """
    Load processed test dataset.

    Expected structure

    data/dataset/test/

        X.npy
        metadata.csv
    """

    test_dir = (
        Path(config.dataset["root"]) /
        config.dataset["test"]
    )

    x_file = test_dir / "X.npy"
    metadata_file = test_dir / "metadata.csv"

    if not x_file.exists():

        raise FileNotFoundError(

            f"Missing dataset:\n{x_file}"

        )

    X = np.load(x_file)

    metadata = None

    if metadata_file.exists():

        metadata = pd.read_csv(

            metadata_file

        )

    return X, metadata


def main():

    print()

    print("=" * 70)

    print("SLEEP TRANSITION INFERENCE")

    print("=" * 70)

    print()

    config = ConfigManager()

    model_path = (

        Path(config.output["root"]) /

        "models" /

        "best_model.keras"

    )

    predictor = Predictor(

        model_path=model_path

    )

    pipeline = InferencePipeline(

        predictor,

        output_dir=(

            Path(config.output["root"]) /

            "predictions"

        )

    )

    print("Loading dataset...")

    X, metadata = load_dataset(

        config

    )

    print(

        f"Loaded {len(X)} samples."

    )

    print()

    results = pipeline.run(

        X,

        metadata

    )

    pipeline.print_summary()

    pipeline.save()

    pipeline.export_summary()

    print()

    print("=" * 70)

    print("INFERENCE COMPLETE")

    print("=" * 70)

    print()

    return results


if __name__ == "__main__":

    main()
