from src.loaders.edf_loader import EDFLoader

loader = EDFLoader(

    preload=False

)

try:
    loader.load(

        r"data/ANPHY/Subjects/EPCTL01/EPCTL01.edf"

    )

    loader.summary()

    print()

    print(

        loader.channel_names[:10]

    )

    print()

    print(

        loader.signal().shape

    )

    print()

    print("EEG Channels")

    print(loader.get_eeg_channels())

    print()

    print("EOG Channels")

    print(loader.get_eog_channels())

    print()

    print("EMG Channels")

    print(loader.get_emg_channels())

    print()

    print("ECG Channels")

    print(loader.get_ecg_channels())

    print()

    print(loader.metadata())
except FileNotFoundError as e:
    print(f"Data not found: {e}. Please ensure the ANPHY dataset is placed in 'data/ANPHY/'.")
