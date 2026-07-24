from src.logging.training_logger import TrainingLogger

logger = TrainingLogger()

logger.training_started(

    model_name="CNN1D",

    epochs=50,

    batch_size=32

)

logger.epoch_finished(

    epoch=1,

    logs={

        "loss":0.512,

        "accuracy":0.84,

        "val_loss":0.47,

        "val_accuracy":0.86

    }

)

logger.checkpoint_saved(

    "outputs/models/best_model.keras"

)

logger.model_saved(

    "outputs/models/final_model.keras"

)

logger.training_finished()
