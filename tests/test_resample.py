from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.sleep_edf import SleepEDFDataset
from src.preprocessing.channel_selector import ChannelSelector
from src.preprocessing.filters import SignalFilter
from src.preprocessing.resample import Resampler


dataset = SleepEDFDataset(Path("data/private"))

record = dataset.load(0)

selector = ChannelSelector()
selector.detect(record)

record.raw = selector.pick_channels(record)
record.signal = record.raw.get_data()

print("Before:", record.raw.info["sfreq"])

SignalFilter().apply(record)
Resampler().apply(record)

print("After :", record.raw.info["sfreq"])
print("Signal Shape:", record.signal.shape)
print("Processing history:")
for entry in record.processing_history:
    print("-", entry)
