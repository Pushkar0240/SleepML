from src.inference.prediction_result import PredictionResult

result = PredictionResult(

    probability=0.91,

    predicted_label=1,

    confidence=0.91,

    threshold=0.50,

    model_name="CNN1D",

    sample_id="001"

)

result.summary()

print()

print(result.to_dict())
