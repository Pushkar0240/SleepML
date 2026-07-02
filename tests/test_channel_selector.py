from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.sleep_edf import SleepEDFDataset
from src.preprocessing.channel_selector import ChannelSelector


dataset = SleepEDFDataset(Path("data/private"))

record = dataset.load(0)

selector = ChannelSelector()

selector.detect(record)

selector.summary()

selected = selector.pick_channels(
    record,
    eeg=True,
    eog=False,
    emg=False,
    ecg=False,
)

print("\nSelected Channels")
print(selected.ch_names)
