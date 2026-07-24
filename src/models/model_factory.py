from src.models.cnn1d import CNN1D
# You can import other models here, e.g. from .cnn import CNN

class ModelFactory:
    """Factory for creating neural network architectures."""
    
    @staticmethod
    def available_models() -> list:
        """Returns a list of available model string identifiers."""
        return ["cnn1d", "cnn"]
        
    @staticmethod
    def create(model_type: str, config=None):
        """
        Creates an instance of the requested model.
        
        Args:
            model_type (str): The architecture to build.
            config: Configuration object containing model params.
            
        Returns:
            BaseModel: An uncompiled model instance inheriting from BaseModel.
        """
        model_type = model_type.lower()
        if model_type == "cnn1d":
            # Extract parameters from config if available, else use defaults
            input_shape = (6000, 7)
            num_classes = 2
            if config:
                try:
                    # attempt to read input shape if available in config
                    pass
                except Exception:
                    pass
            return CNN1D(input_shape=input_shape, num_classes=num_classes)
        # elif model_type == "cnn":
        #    return CNN(...)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
