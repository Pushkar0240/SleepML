from pathlib import Path
import sys


sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data import Recording

record = Recording(
    name="Test Recording",
    dataset="SleepEDF",
    psg_path=Path("dummy.edf"),
)

record.summary()

print("Recording class created successfully.")
