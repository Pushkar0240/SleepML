import numpy as np

from src.inference.predictor import Predictor

predictor = Predictor(

    "outputs/models/best_model.keras"

)

sample = np.random.rand(

    6000,

    7

).astype("float32")

result = predictor.predict(

    sample,

    sample_id="001"

)

result.summary()
