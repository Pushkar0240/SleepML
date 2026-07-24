"""
=========================================================
SleepTransitionFramework

Training Logger

Specialized logger for model training.

Responsibilities
----------------
- Training start/end
- Epoch summaries
- Validation metrics
- Checkpoint events
- Early stopping
- Model saving
- Training duration
=========================================================
"""

from __future__ import annotations

import time

from src.logging import ProjectLogger


class TrainingLogger:
    """
    Logs training-related events.
    """

    # ---------------------------------------------------------

    def __init__(self):

        self.logger = ProjectLogger().get_logger()

        self.start_time = None

    # ---------------------------------------------------------

    def training_started(

        self,

        model_name,

        epochs,

        batch_size

    ):

        self.start_time = time.time()

        self.logger.info("=" * 70)

        self.logger.info("TRAINING STARTED")

        self.logger.info("=" * 70)

        self.logger.info(f"Model      : {model_name}")

        self.logger.info(f"Epochs     : {epochs}")

        self.logger.info(f"Batch Size : {batch_size}")

    # ---------------------------------------------------------

    def epoch_finished(

        self,

        epoch,

        logs

    ):

        loss = logs.get("loss", 0)

        acc = logs.get("accuracy", 0)

        val_loss = logs.get("val_loss", 0)

        val_acc = logs.get("val_accuracy", 0)

        self.logger.info(

            f"Epoch {epoch:03d} | "

            f"Loss={loss:.4f} | "

            f"Accuracy={acc:.4f} | "

            f"Val Loss={val_loss:.4f} | "

            f"Val Accuracy={val_acc:.4f}"

        )

    # ---------------------------------------------------------

    def checkpoint_saved(

        self,

        filepath

    ):

        self.logger.info(

            f"Checkpoint saved -> {filepath}"

        )

    # ---------------------------------------------------------

    def early_stopping(self):

        self.logger.warning(

            "Early stopping triggered."

        )

    # ---------------------------------------------------------

    def model_saved(

        self,

        filepath

    ):

        self.logger.info(

            f"Final model saved -> {filepath}"

        )

    # ---------------------------------------------------------

    def evaluation(

        self,

        accuracy,

        precision,

        recall,

        f1

    ):

        self.logger.info("=" * 70)

        self.logger.info("MODEL EVALUATION")

        self.logger.info("=" * 70)

        self.logger.info(f"Accuracy : {accuracy:.4f}")

        self.logger.info(f"Precision: {precision:.4f}")

        self.logger.info(f"Recall   : {recall:.4f}")

        self.logger.info(f"F1 Score : {f1:.4f}")

    # ---------------------------------------------------------

    def training_finished(self):

        if self.start_time is None:

            return

        duration = time.time() - self.start_time

        hours = int(duration // 3600)

        minutes = int((duration % 3600) // 60)

        seconds = int(duration % 60)

        self.logger.info("=" * 70)

        self.logger.info("TRAINING FINISHED")

        self.logger.info("=" * 70)

        self.logger.info(

            f"Total Time : "

            f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        )

    # ---------------------------------------------------------

    def exception(

        self,

        exception

    ):

        self.logger.exception(exception)
