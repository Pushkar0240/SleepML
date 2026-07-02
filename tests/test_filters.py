from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.sleep_edf import SleepEDFDataset
from src.preprocessing.channel_selector import ChannelSelector
from src.preprocessing.filters import SignalFilter


dataset = SleepEDFDataset(Path("data/private"))

record = dataset.load(0)

selector = ChannelSelector()
selector.detect(record)

record.raw = selector.pick_channels(
    record,
    eeg=True,
    eog=False,
    emg=False,
    ecg=False,
)

record.signal = record.raw.get_data()

print("Before filtering:", record.signal.shape)

filt = SignalFilter()
record = filt.apply(record)

print("After filtering:", record.signal.shape)
print("Processing history:")
for entry in record.processing_history:
    print("-", entry)
