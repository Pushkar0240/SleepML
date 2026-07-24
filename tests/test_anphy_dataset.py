from src.datasets.anphy_dataset import ANPHYDataset

dataset = ANPHYDataset(

    r"data/ANPHY"

)

dataset.summary()

print()

print(

    dataset.subject_ids()

)

print()

print(

    dataset.verify_subject(

        "EPCTL01"

    )

)
