"""
=========================================================
SleepTransitionFramework

Evaluation Script
=========================================================
"""

from pathlib import Path
from src.evaluation.evaluator import ModelEvaluator


from src.config import ConfigManager


def main():

    config = ConfigManager()

    model_path = Path(config.output["root"]) / "models" / "best_model.keras"
    dataset_path = Path(config.dataset["root"]) / config.dataset["test"]

    evaluator = ModelEvaluator(

        model_path=model_path,

        dataset_path=dataset_path

    )

    evaluator.run()


if __name__ == "__main__":

    main()
