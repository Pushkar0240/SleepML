"""
SleepTransitionCNN
==================

Dataset Verifier

Verifies the integrity of the generated dataset (train/val/test splits).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.training.dataset_statistics import DatasetStatistics
from src.logger import logger_factory


logger = logger_factory.get_logger("VerifyDataset")


def verify_split(split_name: str, split_dir: Path):
    logger.info(f"\nVerifying {split_name} split...")
    
    x_path = split_dir / "X.npy"
    y_path = split_dir / "y.npy"
    meta_path = split_dir / "metadata.csv"
    
    if not (x_path.exists() and y_path.exists() and meta_path.exists()):
        logger.error(f"Missing files in {split_name} directory.")
        return False, set()
        
    X = np.load(x_path)
    y = np.load(y_path)
    meta = pd.read_csv(meta_path)
    
    # Check shapes
    if len(X) != len(y) or len(X) != len(meta):
        logger.error(f"Shape mismatch: X({len(X)}), y({len(y)}), meta({len(meta)})")
        return False, set()
        
    # Check NaNs
    if np.isnan(X).any():
        logger.error("WARNING: Found NaNs in X array! Dropping NaNs or manual fix required.")
        return False, set()
        
    if pd.isna(meta["label"]).any():
        logger.error("WARNING: Found missing labels in metadata.")
        return False, set()

    # Get subjects
    subjects = set(meta["subject_id"].unique())
    
    logger.info(f"{split_name} PASS: {len(X)} samples, {len(subjects)} subjects")
    return True, subjects


def verify():
    logger.info("=" * 60)
    logger.info("DATASET VERIFICATION")
    logger.info("=" * 60)
    
    dataset_dir = Path("data/dataset")
    
    splits = ["train", "validation", "test"]
    all_subjects = {}
    
    overall_pass = True
    for split in splits:
        split_dir = dataset_dir / split
        if not split_dir.exists():
            logger.error(f"Split directory {split_dir} does not exist.")
            overall_pass = False
            continue
            
        success, subjects = verify_split(split, split_dir)
        all_subjects[split] = subjects
        if not success:
            overall_pass = False
            
    if len(all_subjects) == 3:
        train_s = all_subjects.get("train", set())
        val_s = all_subjects.get("validation", set())
        test_s = all_subjects.get("test", set())
        
        # Check leakage
        if train_s.intersection(val_s) or train_s.intersection(test_s) or val_s.intersection(test_s):
            logger.error("CRITICAL FAILURE: Data Leakage Detected (Subject overlap between splits).")
            overall_pass = False
        else:
            logger.info("\nPASS: No data leakage detected between splits.")
            
    logger.info("\n" + "=" * 60)
    if overall_pass:
        logger.info("ALL VERIFICATION CHECKS PASSED. Ready for CNN Training.")
    else:
        logger.error("VERIFICATION FAILED. Do not proceed to CNN training.")
    logger.info("=" * 60)


if __name__ == "__main__":
    verify()
