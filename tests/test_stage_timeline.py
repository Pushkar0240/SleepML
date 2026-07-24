from pathlib import Path

from src.datasets.sleep_edf import SleepEDFDataset
from src.labels.hypnogram_parser import HypnogramParser
from src.labels.stage_timeline import StageTimeline

dataset = SleepEDFDataset(
    Path("data/public")
)

record = dataset.load(0)

parser = HypnogramParser()

stages = parser.parse(record)

timeline = StageTimeline()

timeline.build(stages)

timeline.statistics()

timeline.preview(60)
