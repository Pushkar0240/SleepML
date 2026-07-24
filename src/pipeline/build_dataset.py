"""
SleepTransitionCNN
==================

Dataset Build Pipeline

Builds the entire dataset end-to-end.
"""

from pathlib import Path

import pandas as pd

from src.datasets.sleep_edf import SleepEDFDataset
from src.preprocessing.pipeline import PreprocessingPipeline
from src.labels.hypnogram_parser import HypnogramParser
from src.labels.stage_timeline import StageTimeline
from src.labels.transition_detector import TransitionDetector
from src.labels.dataset_builder import DatasetBuilder
from src.labels.export_dataset import DatasetExporter
from src.training.splitter import SubjectSplitter
from src.logger import logger_factory

logger = logger_factory.get_logger("BuildDataset")


def build():
    logger.info("Initializing Dataset Build Pipeline...")

    dataset_path = Path("data/public")
    if not dataset_path.exists():
        logger.error(f"Dataset path {dataset_path} does not exist.")
        return

    dataset = SleepEDFDataset(dataset_path)
    
    # We will aggregate samples and metadata across all recordings
    all_samples = []
    
    # Process the first few recordings (e.g. 2 for demonstration, but typically all)
    # Since we need enough subjects to split, we will load a few if available.
    try:
        record_indices = range(dataset.__len__()) if hasattr(dataset, '__len__') else [0, 1]
    except Exception:
        record_indices = [0]

    for idx in record_indices:
        try:
            logger.info(f"Processing recording {idx}...")
            record = dataset.load(idx)
            
            # 1. Preprocessing
            record = PreprocessingPipeline().process(record)
            
            # 2. Parse Hypnogram & Timeline
            stages = HypnogramParser().parse(record)
            timeline = StageTimeline().build(stages)
            
            # 3. Detect Transitions
            transitions = TransitionDetector(window_size=30).detect(timeline)
            
            # 4. Generate Windows (Samples)
            positive, negative = DatasetBuilder().build(record, transitions)
            
            all_samples.extend(positive)
            all_samples.extend(negative)
        except Exception as e:
            logger.error(f"Failed to process recording {idx}: {e}")

    if not all_samples:
        logger.error("No samples were generated.")
        return

    # Create metadata DataFrame
    logger.info("Creating full metadata DataFrame...")
    
    # We need to build the dataframe in the format expected by SubjectSplitter and MetadataManager
    metadata_rows = []
    for sample in all_samples:
        metadata_rows.append({
            "sample_id": sample.get("sample_id"),
            "subject_id": sample.get("subject"),
            "recording_id": sample.get("recording"),
            "dataset": sample.get("dataset"),
            "label": 1 if sample.get("label") == "Transition" else 0,
            "transition_type": sample.get("transition"),
            "start_second": sample.get("start_sec"),
            "end_second": sample.get("end_sec"),
            "sampling_rate": 100, # default
        })
        
    metadata_df = pd.DataFrame(metadata_rows)
    
    # 5. Split Dataset by Subject
    logger.info("Splitting dataset by subject...")
    splitter = SubjectSplitter()
    train_df, val_df, test_df = splitter.split(metadata_df)
    
    # Export splits
    splitter.export(train_df, val_df, test_df, output_folder="data/dataset")
    
    # 6. Export X.npy, y.npy, and metadata for each split
    logger.info("Exporting actual samples (X.npy, y.npy, metadata.csv)...")
    exporter = DatasetExporter()
    exporter.export_split(all_samples, train_df, output_dir="data/dataset/train")
    exporter.export_split(all_samples, val_df, output_dir="data/dataset/validation")
    exporter.export_split(all_samples, test_df, output_dir="data/dataset/test")
    
    logger.info("Dataset Pipeline completed successfully!")


if __name__ == "__main__":
    build()
