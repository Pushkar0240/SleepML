from src.config.config_loader import ConfigLoader

config = ConfigLoader()

config.summary()

print()

print("Learning Rate")

print(

    config.get(

        "training",

        "learning_rate"

    )

)

print()

config.set(

    "training",

    "epochs",

    value=100

)

print(

    "Updated Epochs:",

    config.get(

        "training",

        "epochs"

    )

)

print()

print("Sections")

print(

    config.sections()

)
