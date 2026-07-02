from pathlib import Path
import sys


sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.datasets.base_dataset import BaseDataset

print("BaseDataset imported successfully.")
