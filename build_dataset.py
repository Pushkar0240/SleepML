"""
=========================================================
SleepTransitionCNN

Dataset Builder Pipeline

Pipeline

SleepEDF
    ↓
Recording
    ↓
Preprocessing
    ↓
Hypnogram Parser
    ↓
Transition Detection
    ↓
Window Generator
    ↓
TrainingSample
    ↓
Dataset Export
=========================================================
"""

from pathlib import Path
import traceback

from src.datasets.sleep_edf import SleepEDFDataset
from src.preprocessing.pipeline import PreprocessingPipeline
from src.labels.hypnogram_parser import HypnogramParser
from src.labels.stage_timeline import StageTimeline
from src.labels.transition_detector import TransitionDetector
from src.labels.dataset_builder import DatasetBuilder

from src.labels.export_dataset import DatasetExporter
from src.training.metadata import MetadataManager

# ---------------------------------------------------------

DATASET_PATH = Path("data/public")

WINDOW_SIZE = 30

# ---------------------------------------------------------


def validate_samples(samples):

    """
    Ensure all windows have identical shape.
    """

    if len(samples) == 0:
        raise RuntimeError("No training samples generated.")

    reference = samples[0].signal.shape

    for sample in samples:

        if sample.signal.shape != reference:

            raise ValueError(

                "Inconsistent window shape detected.\n"

                f"Expected : {reference}\n"

                f"Found    : {sample.signal.shape}"

            )

    print()

    print("Dataset Validation Passed")

    print(f"Window Shape : {reference}")

    print(f"Samples : {len(samples)}")


# ---------------------------------------------------------


def main():

    print("=" * 80)
    print("SleepTransitionCNN Dataset Builder")
    print("=" * 80)

    dataset = SleepEDFDataset(DATASET_PATH)

    preprocessing = PreprocessingPipeline()

    parser = HypnogramParser()

    detector = TransitionDetector(

        window_size=WINDOW_SIZE

    )

    builder = DatasetBuilder(

        window_size=WINDOW_SIZE

    )

    exporter = DatasetExporter()

    metadata = MetadataManager()

    all_samples = []

    print()

    print(f"Recordings Found : {len(dataset)}")

    print()

    # -----------------------------------------------------

    for index in range(len(dataset)):

        print("=" * 80)

        print(

            f"Recording {index+1}/{len(dataset)}"

        )

        print("=" * 80)

        try:

            # --------------------------------------------

            record = dataset.load(index)

            # --------------------------------------------

            preprocessing.process(record)

            # --------------------------------------------

            stages = parser.parse(record)

            # --------------------------------------------

            timeline = StageTimeline().build(

                stages

            )

            # --------------------------------------------

            transitions = detector.detect(

                timeline

            )

            # --------------------------------------------

            samples = builder.build(

                record,

                transitions

            )

            all_samples.extend(samples)

            print()

            print(

                f"Generated Samples : {len(samples)}"

            )

            print()

        except Exception:

            traceback.print_exc()

            print()

    # -----------------------------------------------------

    print()

    print("=" * 80)

    print("Final Validation")

    print("=" * 80)

    validate_samples(all_samples)

    # -----------------------------------------------------

    print()

    print("=" * 80)

    print("Exporting Dataset")

    print("=" * 80)

    exporter.export(

        all_samples

    )

    metadata.export(

        all_samples

    )

    print()

    print("=" * 80)

    print("DATASET BUILD COMPLETED")

    print("=" * 80)

    print()

    print(f"Total Samples : {len(all_samples)}")

    print()


# ---------------------------------------------------------

if __name__ == "__main__":

    main()
