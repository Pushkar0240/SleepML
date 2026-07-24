import numpy as np

from src.inference.predictor import Predictor
from src.inference.inference_pipeline import InferencePipeline

predictor = Predictor(

    "outputs/models/best_model.keras"

)

pipeline = InferencePipeline(

    predictor

)

X = np.random.rand(

    10,

    6000,

    7

).astype("float32")

results = pipeline.run(X)

print(results.head())

pipeline.print_summary()

pipeline.save()

pipeline.export_summary()
