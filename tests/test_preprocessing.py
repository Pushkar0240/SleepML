from pathlib import Path

from src.datasets.sleep_edf import SleepEDFDataset
from src.preprocessing.pipeline import PreprocessingPipeline

dataset = SleepEDFDataset(

    Path("data/public")

)

record = dataset.load(0)

pipeline = PreprocessingPipeline()

record = pipeline.process(record)

record.summary()

print()

print("History")

for step in record.processing_history:

    print("-", step)
