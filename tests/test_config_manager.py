from src.config import ConfigManager

config = ConfigManager()

print()

print("=" * 60)

print("CONFIG MANAGER TEST")

print("=" * 60)

print()

print(

    "Project :",

    config.project["name"]

)

print(

    "Version :",

    config.project["version"]

)

print()

print(

    "Model :",

    config.model["name"]

)

print(

    "Epochs :",

    config.training["epochs"]

)

print(

    "Batch Size :",

    config.training["batch_size"]

)

print(

    "Learning Rate :",

    config.training["learning_rate"]

)

print()

print(

    "Dataset Root :",

    config.dataset["root"]

)

print()

config.summary()
