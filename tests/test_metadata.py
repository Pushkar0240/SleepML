from src.training.metadata import (
    MetadataManager,
    SampleMetadata
)

manager = MetadataManager()

manager.add(

    SampleMetadata(

        sample_id=1,

        subject_id="SC4001",

        recording_id="SC4001E0",

        dataset="SleepEDF",

        label=1,

        transition_type="Wake -> N1",

        previous_stage="Wake",

        next_stage="N1",

        start_second=3030,

        center_second=3060,

        end_second=3090,

        sampling_rate=100,

        channels="EEG Fpz-Cz, EEG Pz-Oz"

    )

)

manager.summary()

manager.export()
