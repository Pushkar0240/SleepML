"""Project entry point for the SleepEDF dataset demo."""

from src.config import config
from src.datasets import SleepEDFDataset


def main() -> None:
    dataset_root = config.private_data / "edf"
    dataset = SleepEDFDataset(dataset_root)

    if dataset.number_of_recordings() == 0:
        raise FileNotFoundError(
            f"No Sleep-EDF PSG/Hypnogram pairs found in {dataset_root.resolve()}"
        )

    dataset.summary()
    dataset.print_recordings()

    choice = int(input("\nSelect recording number: "))

    record = dataset.load(choice - 1)
    record.summary()


if __name__ == "__main__":
    main()