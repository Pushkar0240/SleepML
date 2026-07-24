"""
=========================================================
SleepTransitionFramework

Model Evaluator

Evaluates trained models on the test dataset.
=========================================================
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)

from src.training.dataloader import EEGDataLoader


class ModelEvaluator:

    def __init__(

        self,

        model_path=Path("outputs/models/best_model.keras"),

        dataset_path="data/dataset/test",

        output_dir="outputs"

    ):

        self.model_path = Path(model_path)

        self.dataset_path = Path(dataset_path)

        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(

            parents=True,

            exist_ok=True

        )

    # -----------------------------------------------------

    def load(self):

        print("\nLoading model...")

        self.model = tf.keras.models.load_model(

            self.model_path

        )

        loader = EEGDataLoader(

            self.dataset_path,

            batch_size=32,

            shuffle=False

        )

        self.dataset = loader.dataset()

        self.X, self.y = loader.load_numpy()

    # -----------------------------------------------------

    def predict(self):

        print("Generating predictions...")

        probabilities = self.model.predict(

            self.dataset,

            verbose=0

        )

        self.probabilities = probabilities.flatten()

        self.predictions = (

            self.probabilities >= 0.5

        ).astype(int)

    # -----------------------------------------------------

    def evaluate(self):

        print("\nEvaluation Results")

        print("-" * 50)

        accuracy = accuracy_score(

            self.y,

            self.predictions

        )

        auc = roc_auc_score(

            self.y,

            self.probabilities

        )

        print(f"Accuracy : {accuracy:.4f}")

        print(f"AUC      : {auc:.4f}")

        print()

        print(

            classification_report(

                self.y,

                self.predictions,

                digits=4

            )

        )

    # -----------------------------------------------------

    def confusion(self):

        cm = confusion_matrix(

            self.y,

            self.predictions

        )

        disp = ConfusionMatrixDisplay(

            confusion_matrix=cm

        )

        disp.plot()

        plt.tight_layout()

        plt.savefig(

            self.output_dir /

            "confusion_matrix.png",

            dpi=300

        )

        plt.close()

    # -----------------------------------------------------

    def roc(self):

        fpr, tpr, _ = roc_curve(

            self.y,

            self.probabilities

        )

        plt.figure(figsize=(6,6))

        plt.plot(

            fpr,

            tpr,

            label="ROC"

        )

        plt.plot(

            [0,1],

            [0,1],

            "--"

        )

        plt.xlabel("False Positive Rate")

        plt.ylabel("True Positive Rate")

        plt.title("ROC Curve")

        plt.grid(True)

        plt.legend()

        plt.tight_layout()

        plt.savefig(

            self.output_dir /

            "roc_curve.png",

            dpi=300

        )

        plt.close()

    # -----------------------------------------------------

    def precision_recall(self):

        precision, recall, _ = precision_recall_curve(

            self.y,

            self.probabilities

        )

        plt.figure(figsize=(6,6))

        plt.plot(

            recall,

            precision

        )

        plt.xlabel("Recall")

        plt.ylabel("Precision")

        plt.title("Precision-Recall Curve")

        plt.grid(True)

        plt.tight_layout()

        plt.savefig(

            self.output_dir /

            "precision_recall_curve.png",

            dpi=300

        )

        plt.close()

    # -----------------------------------------------------

    def run(self):

        self.load()

        self.predict()

        self.evaluate()

        self.confusion()

        self.roc()

        self.precision_recall()

        print()

        print("Evaluation completed.")
