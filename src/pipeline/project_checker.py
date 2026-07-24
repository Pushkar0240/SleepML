"""
SleepTransitionCNN
==================

Project Checker

Verifies the workspace and environment before running the dataset generation pipeline.
"""

import os
import sys
from pathlib import Path


def check_dependencies():
    print("Checking dependencies...")
    missing = []
    
    try:
        import mne
    except ImportError:
        missing.append("mne")
        
    try:
        import pandas as pd
    except ImportError:
        missing.append("pandas")
        
    try:
        import tensorflow as tf
    except ImportError:
        missing.append("tensorflow")
        
    try:
        import numpy as np
    except ImportError:
        missing.append("numpy")
        
    try:
        import sklearn
    except ImportError:
        missing.append("scikit-learn")

    if missing:
        print(f"FAILED: Missing dependencies: {', '.join(missing)}")
        print(f"Please install them using: pip install {' '.join(missing)}")
        return False
    
    print("PASS: All Python dependencies found.")
    return True


def check_directories():
    print("\nChecking directories...")
    directories = [
        "data/public",
        "data/dataset",
        "outputs",
        "models"
    ]
    
    all_exist = True
    for directory in directories:
        path = Path(directory)
        if not path.exists():
            print(f"Creating missing directory: {directory}")
            path.mkdir(parents=True, exist_ok=True)
            
    print("PASS: All directories are ready.")
    return True


def check_project():
    print("=" * 60)
    print("PROJECT CHECKER")
    print("=" * 60)
    
    deps_ok = check_dependencies()
    dirs_ok = check_directories()
    
    print("=" * 60)
    if deps_ok and dirs_ok:
        print("ALL CHECKS PASSED. Ready to build dataset.")
        return True
    else:
        print("WARNING: Some checks failed. Please fix before continuing.")
        return False


if __name__ == "__main__":
    check_project()
