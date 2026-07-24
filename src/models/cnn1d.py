from tensorflow.keras import layers, models
from src.models.base_model import BaseModel

class CNN1D(BaseModel):
    """
    1D CNN architecture for Sleep Stage Transition Detection.
    Inherits from BaseModel.
    """
    
    def __init__(self, input_shape=(6000, 2), num_classes=2):
        super().__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        
    def build(self):
        """Builds the Keras model based on the README specs."""
        inputs = layers.Input(shape=self.input_shape)
        
        # Stage 1
        x = layers.Conv1D(filters=32, kernel_size=7, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        
        # Stage 2
        x = layers.Conv1D(filters=64, kernel_size=5, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        
        # Stage 3
        x = layers.Conv1D(filters=128, kernel_size=3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        
        # Global Average Pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        # Dense hidden layer with Dropout
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        
        # Output layer (Binary Classification)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        self.model = models.Model(inputs=inputs, outputs=outputs, name='SleepTransitionCNN1D')
        return self.model
