"""
=========================================================
SleepTransitionFramework

EDF Loader

Production-quality EDF+ loader for the ANPHY Sleep Dataset.

Features
--------
- Load EDF+
- Read metadata
- Read channel names
- Sampling frequency
- Recording duration
- Channel selection
- Crop recordings
- Export signals
- Signal type identification (EEG, EOG, EMG, ECG)

Author
------
Pushkar Singh
=========================================================
"""

from __future__ import annotations

from pathlib import Path

import mne
import numpy as np


class EDFLoader:
    """
    Production EDF loader.
    """

    # -----------------------------------------------------

    def __init__(

        self,

        preload=False,

        verbose=False

    ):

        self.preload = preload

        self.verbose = verbose

        self.raw = None

    # -----------------------------------------------------

    def load(

        self,

        filename

    ):

        """
        Load EDF recording.
        """

        filename = Path(filename)

        if not filename.exists():

            raise FileNotFoundError(

                filename

            )

        self.raw = mne.io.read_raw_edf(

            filename,

            preload=self.preload,

            verbose=self.verbose

        )

        return self.raw

    # -----------------------------------------------------

    @property
    def channel_names(self):

        return self.raw.ch_names

    # -----------------------------------------------------

    @property
    def number_of_channels(self):

        return len(

            self.raw.ch_names

        )

    # -----------------------------------------------------

    @property
    def sampling_frequency(self):

        return int(

            self.raw.info["sfreq"]

        )

    # -----------------------------------------------------

    @property
    def duration(self):

        return self.raw.times[-1]

    # -----------------------------------------------------

    @property
    def recording_info(self):

        return self.raw.info

    # -----------------------------------------------------

    @property
    def start_time(self):

        return self.raw.info["meas_date"]

    # -----------------------------------------------------
    # Channel Type Utilities
    # -----------------------------------------------------

    def channels_by_type(self):
        """
        Return channels grouped by signal type.

        Returns
        -------
        dict
            {
                "eeg": [...],
                "eog": [...],
                "emg": [...],
                "ecg": [...],
                "misc": [...]
            }
        """

        groups = {
            "eeg": [],
            "eog": [],
            "emg": [],
            "ecg": [],
            "misc": []
        }

        channel_types = self.raw.get_channel_types()

        for name, ctype in zip(self.channel_names, channel_types):

            if ctype in groups:
                groups[ctype].append(name)
            else:
                groups["misc"].append(name)

        return groups

    # -----------------------------------------------------

    def get_eeg_channels(self):

        return self.channels_by_type()["eeg"]

    # -----------------------------------------------------

    def get_eog_channels(self):

        return self.channels_by_type()["eog"]

    # -----------------------------------------------------

    def get_emg_channels(self):

        return self.channels_by_type()["emg"]

    # -----------------------------------------------------

    def get_ecg_channels(self):

        return self.channels_by_type()["ecg"]

    # -----------------------------------------------------

    def get_misc_channels(self):

        return self.channels_by_type()["misc"]

    # -----------------------------------------------------
    # Pick Signal Types
    # -----------------------------------------------------

    def pick_eeg(self):

        return self.raw.copy().pick(
            self.get_eeg_channels()
        )

    # -----------------------------------------------------

    def pick_eog(self):

        return self.raw.copy().pick(
            self.get_eog_channels()
        )

    # -----------------------------------------------------

    def pick_emg(self):

        return self.raw.copy().pick(
            self.get_emg_channels()
        )

    # -----------------------------------------------------

    def pick_ecg(self):

        return self.raw.copy().pick(
            self.get_ecg_channels()
        )

    # -----------------------------------------------------
    # Metadata
    # -----------------------------------------------------

    def metadata(self):

        return {

            "channels": self.number_of_channels,

            "sampling_frequency": self.sampling_frequency,

            "duration_seconds": self.duration,

            "start_time": self.start_time,

            "channel_names": self.channel_names,

            "channel_types": self.raw.get_channel_types()

        }

    # -----------------------------------------------------

    def signal(

        self,

        picks=None

    ):

        """
        Return signal as NumPy array.

        Shape

        channels × samples
        """

        data = self.raw.get_data(

            picks=picks

        )

        return data

    # -----------------------------------------------------

    def dataframe(

        self,

        picks=None

    ):

        return self.raw.to_data_frame(

            picks=picks

        )

    # -----------------------------------------------------

    def channel_exists(

        self,

        channel

    ):

        return channel in self.channel_names

    # -----------------------------------------------------

    def select_channels(

        self,

        channels

    ):

        """
        Return copied Raw object.
        """

        return self.raw.copy().pick(

            channels

        )

    # -----------------------------------------------------

    def crop(

        self,

        tmin,

        tmax

    ):

        """
        Crop recording.
        """

        return self.raw.copy().crop(

            tmin=tmin,

            tmax=tmax

        )

    # -----------------------------------------------------

    def time_vector(self):

        return self.raw.times

    # -----------------------------------------------------

    def summary(self):

        print()

        print("=" * 70)

        print("EDF SUMMARY")

        print("=" * 70)

        print()

        print(

            "Channels :",

            self.number_of_channels

        )

        print(

            "Sampling Frequency :",

            self.sampling_frequency,

            "Hz"

        )

        print(

            "Duration :",

            f"{self.duration:.2f}",

            "seconds"

        )

        print(

            "Start Time :",

            self.start_time

        )

        print()

        print("=" * 70)

    # -----------------------------------------------------

    def __len__(self):

        return self.number_of_channels

    # -----------------------------------------------------

    def __repr__(self):

        if self.raw is None:

            return "EDFLoader(not loaded)"

        return (

            f"EDFLoader("

            f"{self.number_of_channels} channels, "

            f"{self.sampling_frequency} Hz)"

        )
