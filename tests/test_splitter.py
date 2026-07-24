import pandas as pd

from src.training.splitter import SubjectSplitter


metadata = pd.DataFrame({

    "subject_id": [

        "S1",

        "S1",

        "S2",

        "S2",

        "S3",

        "S4",

        "S5",

        "S6",

        "S7",

        "S8"

    ],

    "sample": list(range(10))

})

splitter = SubjectSplitter()

train_df, val_df, test_df = splitter.split(

    metadata

)

splitter.export(

    train_df,

    val_df,

    test_df

)

print()

print(train_df)

print()

print(val_df)

print()

print(test_df)
