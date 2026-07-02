from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.sleep_edf import SleepEDFDataset
from src.labels.hypnogram_parser import HypnogramParser


dataset = SleepEDFDataset(
    Path("data/private")
)

record = dataset.load(0)

parser = HypnogramParser()

parser.parse(record)

parser.summary()
