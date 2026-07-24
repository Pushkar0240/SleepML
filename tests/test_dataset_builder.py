from pathlib import Path

from src.datasets.sleep_edf import SleepEDFDataset

from src.preprocessing.pipeline import PreprocessingPipeline

from src.labels.hypnogram_parser import HypnogramParser

from src.labels.stage_timeline import StageTimeline

from src.labels.transition_detector import TransitionDetector

from src.labels.dataset_builder import DatasetBuilder

from src.labels.export_dataset import DatasetExporter


dataset = SleepEDFDataset(
    Path("data/public")
)

record = dataset.load(0)

record = PreprocessingPipeline().process(record)

stages = HypnogramParser().parse(record)

timeline = StageTimeline().build(stages)

transitions = TransitionDetector().detect(timeline)

positive, negative = DatasetBuilder().build(
    record,
    transitions
)

DatasetExporter().export(
    positive,
    negative
)
