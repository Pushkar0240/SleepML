from pathlib import Path

from src.datasets.sleep_edf import SleepEDFDataset

from src.labels.hypnogram_parser import HypnogramParser

from src.labels.stage_timeline import StageTimeline

from src.labels.transition_detector import TransitionDetector


dataset = SleepEDFDataset(
    Path("data/public")
)

record = dataset.load(0)

stages = HypnogramParser().parse(record)

timeline = StageTimeline().build(stages)

detector = TransitionDetector(window_size=30)

detector.detect(timeline)

detector.summary()
