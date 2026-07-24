import numpy as np

from src.inference.predictor import Predictor
from src.inference.batch_predictor import BatchPredictor

predictor = Predictor(

    "outputs/models/best_model.keras"

)

batch = BatchPredictor(

    predictor

)

X = np.random.rand(

    20,

    6000,

    7

).astype("float32")

batch.predict_array(X)

batch.print_summary()

batch.save()
