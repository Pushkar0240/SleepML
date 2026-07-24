"""
=========================================================
SleepTransitionFramework

Run Manager

Coordinates the complete lifecycle of a training run.

Responsibilities
----------------
- Load configuration
- Create experiment
- Register experiment
- Create model
- Train model
- Save results
- Update experiment status
=========================================================
"""

from __future__ import annotations

from src.config import ConfigManager
from src.experiments import (
    ExperimentManager,
    ExperimentTracker,
)
from src.logging.training_logger import TrainingLogger
from src.models import ModelFactory
from src.training.trainer import ModelTrainer


class RunManager:
    """
    High-level manager for training experiments.
    """

    # ---------------------------------------------------------

    def __init__(self):

        self.config = ConfigManager()

        self.experiment = ExperimentManager()

        self.tracker = ExperimentTracker()

        self.logger = TrainingLogger()

    # ---------------------------------------------------------

    def setup(self):

        """
        Prepare experiment workspace.
        """

        self.experiment.create()

        self.tracker.register(

            self.experiment.experiment_id,

            self.config

        )

    # ---------------------------------------------------------

    def create_model(self):

        """
        Create model using ModelFactory.
        """

        model = ModelFactory.create(

            self.config.model["name"],
            
            config=self.config

        )

        return model

    # ---------------------------------------------------------

    def train(self):

        """
        Execute training.
        """

        model = self.create_model()

        self.logger.training_started(

            model_name=self.config.model["name"],

            epochs=self.config.training["epochs"],

            batch_size=self.config.training["batch_size"]

        )

        trainer = ModelTrainer(

            model=model.get_model(),

            config=self.config,
            
            experiment=self.experiment

        )

        history = trainer.train()

        trainer.save(

            self.experiment.models_dir()

            / "final_model.keras"

        )

        self.logger.training_finished()

        self.tracker.update_status(

            self.experiment.experiment_id,

            "Completed"

        )

        return history

    # ---------------------------------------------------------

    def failed(

        self,

        exception

    ):

        """
        Handle failed experiment.
        """

        self.logger.exception(exception)

        self.tracker.update_status(

            self.experiment.experiment_id,

            "Failed"

        )

    # ---------------------------------------------------------

    def run(self):

        """
        Execute complete experiment.
        """

        try:

            self.setup()

            history = self.train()

            return history

        except Exception as error:

            self.failed(error)

            raise
