from src.experiments import ExperimentManager

experiment = ExperimentManager()

experiment.create()

experiment.summary()

print()

print(

    "Models :", experiment.models_dir()

)

print(

    "Logs :", experiment.logs_dir()

)

print(

    "History :", experiment.history_dir()

)

print(

    "Evaluation :", experiment.evaluation_dir()

)

print(

    "Plots :", experiment.plots_dir()

)

print(

    "Predictions :", experiment.predictions_dir()

)
