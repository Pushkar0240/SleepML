from src.models import ModelFactory

print()

print("Available Models")

print(

    ModelFactory.available_models()

)

print()

model = ModelFactory.create(

    "cnn1d"

)

model.summary()
