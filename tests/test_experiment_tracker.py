from src.config import ConfigManager
from src.experiments import ExperimentManager
from src.experiments.experiment_tracker import ExperimentTracker

config = ConfigManager(validate=False)

experiment = ExperimentManager()

experiment.create()

tracker = ExperimentTracker()

tracker.register(

    experiment.experiment_id,

    config

)

tracker.update_status(

    experiment.experiment_id,

    "Completed"

)

tracker.update_metrics(

    experiment.experiment_id,

    accuracy=0.92,

    precision=0.91,

    recall=0.90,

    f1=0.905,

    auc=0.96

)

tracker.summary()
