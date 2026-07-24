# SleepTransitionCNN
Deep learning pipeline for detecting sleep stage transitions using raw sleep EEG recordings.

## Overview
This repository contains a deep learning pipeline designed to detect sleep stage transitions (e.g., sleep-onset, awakening, micro-arousals) directly from raw EEG signals rather than relying on rigid 30-second epoch classifications. The project processes multi-channel polysomnography (PSG) data and hypnogram annotations from the Sleep-EDF database (with built-in support for MASS, ISRUC, or private datasets) to extract windows around transition boundaries. It then trains and evaluates a 1D Convolutional Neural Network (CNN) to classify whether an EEG window represents a stable sleep stage or a sleep stage transition.

## Background / Motivation
Standard sleep staging (polysomnography) divides EEG recordings into discrete, non-overlapping 30-second epochs. While useful for general sleep architectures, this arbitrary classification scheme masks the exact transient moments when the brain shifts from one state to another (such as transitioning from wakefulness to N1 sleep, or sudden micro-arousals). By treating transition detection as a continuous signal processing and binary classification task on raw time-series data, this project aims to bypass the constraints of fixed epoch boundaries and offer a more precise window into sleep dynamics.

## Data
- **Source & Format:** The pipeline is designed around the public [Sleep-EDF Database](https://physionet.org/content/sleep-edf/) which provides PSG recordings and hypnogram annotations in EDF format.
- **Channels:** The pipeline automatically identifies and loads preferred EEG channels (e.g., `EEG Fpz-Cz`, `EEG Pz-Oz`, `C3-A2`, `C4-A1`, `Fpz-Cz`) matching what is available in the recording.
- **Preprocessing:** Raw signals are resampled to a target frequency (defaults to 100 Hz), bandpass filtered (default `0.3` to `35` Hz), notch filtered (default `50` Hz), and optionally normalized using Z-score standardization.
- **Window Generation:** EEG windows are generated centered on detected sleep transitions. For a 30-second window size configuration, this extracts 30 seconds before and 30 seconds after the transition center (yielding a 60-second segment, or 6,000 time steps at 100 Hz).
- **Negative (Stable) Sampling:** Negative samples are randomly drawn from periods that do not overlap with any sleep transitions within the configured window size margin.
- **Supply Your Own Data:** Raw files are not included in this repository. To run the pipeline, supply your own `.edf` recordings by placing them in the following directory:
  ```text
  data/public/
  ```
  Ensure your PSG recordings end with `PSG.edf` and annotations end with `Hypnogram.edf` (e.g., `SC4001E0-PSG.edf` and `SC4001EC-Hypnogram.edf`).

## Installation
- **Python Version:** Python 3.8+ is recommended.
- **Dependency Installation:** Make sure you are inside your virtual environment and run the following command to install the required Python packages:
  ```bash
  pip install mne pandas tensorflow numpy scikit-learn pyyaml
  ```
  *(Note: The `requirements.txt` file in the repository root currently tracks `mne`. Ensure all of the above dependencies are installed to avoid imports failing).*

## Project Structure
Below is the directory tree of the repository:

```text
SleepTransitionCNN/
├── build_dataset.py       # Main entry point to run the dataset generation pipeline
├── verify_dataset.py      # Script to verify the structure, labels, and integrity of the exported dataset
├── config.yaml            # Central configuration file for hyperparameters and paths
├── main.py                # Legacy/demo entry point for loading Sleep-EDF dataset
├── train.py               # Top-level script to trigger the CNN training loop
├── requirements.txt       # Python package dependencies
├── src/                   # Source code
│   ├── config.py          # Configuration manager parsing config.yaml
│   ├── data/              # Data containers (Recording, TrainingSample)
│   │   ├── recording.py
│   │   └── training_sample.py
│   ├── datasets/          # Dataset loaders (SleepEDF, MASS, ISRUC, etc.)
│   │   ├── base_dataset.py
│   │   └── sleep_edf.py
│   ├── labels/            # Dataset parsing and window generation modules
│   │   ├── dataset_builder.py
│   │   ├── export_dataset.py
│   │   ├── hypnogram_parser.py
│   │   ├── stage_timeline.py
│   │   ├── transition_detector.py
│   │   └── window_generator.py
│   ├── models/            # Deep learning architectures
│   │   ├── cnn1d.py       # Keras Sequential 1D CNN model
│   │   └── losses.py
│   ├── pipeline/          # Integration pipelines
│   │   ├── project_checker.py # Workspace and dependencies validation utility
│   │   └── verify_dataset.py  # Pipeline verifier duplicate/legacy
│   ├── preprocessing/     # Signal processing modules
│   │   ├── channel_selector.py
│   │   ├── filters.py
│   │   ├── normalize.py
│   │   ├── pipeline.py    # Main PreprocessingPipeline runner
│   │   └── resample.py
│   ├── training/          # Training infrastructure
│   │   ├── balance.py     # Class balancer (oversample, undersample, class weights)
│   │   ├── dataloader.py  # tf.data.Dataset loader and shape transposer
│   │   ├── split_dataset.py # Subject-wise train/val/test data partitioner
│   │   └── trainer.py     # Training execution logic
│   └── utils/             # Logging and utility helpers
│       └── logger.py
└── tests/                 # Unit and integration test suite
```

## Usage

### 1. Verify Environment
Verify that your directory structure and dependencies are ready:
```bash
python src/pipeline/project_checker.py
```

### 2. Generate Dataset
Build the training dataset from the raw EDF files in `data/public/`:
```bash
python build_dataset.py
```

### 3. Verify Dataset
Run the data verifier to check for NaN values, shape consistency, and correct label balance:
```bash
python verify_dataset.py
```

### 4. Create Subject-wise Splits
Partition the generated dataset into subject-wise train/validation/test folders:
```bash
python src/training/split_dataset.py
```

### 5. Train the 1D CNN
Execute the training loop to fit the model:
```bash
python train.py
```

## Methodology Summary
1. **Preprocessing Pipeline:** Resamples the input EEG signal to 100 Hz. Applies a bandpass filter (0.3 - 35 Hz) and a notch filter (50 Hz) using `mne` to isolate relevant cortical activity and reduce powerline noise.
2. **Transition Parsing:** The `HypnogramParser` converts hypnogram events into a second-by-second stages array, and `TransitionDetector` identifies timestamps where stages change (e.g. Wake to N1).
3. **Window Extraction:** Extract windows around transition boundaries. If a 30s window size is set, it extracts `WINDOW_SIZE * 2 * sampling_rate` points. For 30s windows at 100 Hz, the window size is `6,000` samples per channel.
4. **Data Balancing:** Stable/negative sample seconds are generated dynamically using `overlaps_transition` to ensure they are at least `self.window_size` seconds away from any transition. Classes can be balanced via oversampling or undersampling methods in `DatasetBalancer`.
5. **DataLoader Transposition:** The `EEGDataLoader` transposes datasets in-memory from disk format `(samples, channels, time)` to `(samples, time, channels)` in memory so they can be fed into `tf.keras.layers.Conv1D`.
6. **CNN Architecture:** A 1D CNN architecture composed of:
   - Three 1D Convolution stages (filters: 32, 64, 128; kernel sizes: 7, 5, 3) with `BatchNormalization` and `MaxPooling1D`.
   - Global average pooling (`GlobalAveragePooling1D`).
   - Dense hidden layer (64 units, ReLU).
   - Dropout layer (rate of 0.5) for regularization.
   - Dense output layer (2 units, Softmax activation) predicting stable (0) vs. transition (1).

## Outputs
- **Dataset Arrays:** Saved to `data/dataset/` containing:
  - `X.npy`: The multi-channel raw EEG windows.
  - `y.npy`: Binary labels (0 = Stable, 1 = Transition).
  - `metadata.csv`: Traceability mapping mapping window IDs to their origin subject, recording, window times, and stage transition metadata.
- **Subject-wise Folders:** Partitions created inside `data/dataset/train/`, `data/dataset/validation/`, and `data/dataset/test/` respectively.
- **Model Checkpoints:** Trained models are automatically saved as `models/cnn1d.keras` using the validation loss callback.

## Status / Limitations
- **Work in Progress:** This repository is an active research project.
- **Sample Size:** The dataset splits are currently highly dependent on the number of raw EDF files placed inside `data/public/`. (During local development, a single subject `SC4001` yielded 304 samples, requiring a larger pool of subjects for standard subject-wise splitting).
- **Artifact Rejection:** Currently lacks a dedicated automatic artifact rejection step (e.g., ICA or threshold-based artifact rejection).
- [TODO: list any other limitations if found in subsequent model evaluation].

## References
- Sleep-EDF Database: [PhysioNet Sleep-EDF Database Expanded](https://physionet.org/content/sleep-edf/)
- [TODO: Add any academic papers, lab publications, or citations associated with the underlying project/thesis]

## Author / Acknowledgments

- **Acknowledgments:** [TODO: fill in acknowledgments or funding sources]
