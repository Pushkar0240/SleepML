from pathlib import Path
import sys


sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.datasets.sleep_edf import SleepEDFDataset

DATASET = SleepEDFDataset(
    Path("data/private")
)

DATASET.summary()

DATASET.print_recordings()

record = DATASET.load(0)

record.summary()
