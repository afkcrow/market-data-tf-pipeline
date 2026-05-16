"""
src/models/lstm_forecaster.py
Updated LSTM model that automatically detects number of features.
"""

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import pandas as pd
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LSTMForecaster:
    """LSTM model that adapts to the actual number of features in the data."""

    def __init__(self, seq_length: int = 60, units: int = 64):
        self.seq_length = seq_length
        self.units = units
        self.n_features: Optional[int] = None   # Will be set dynamically
        self.model: Optional[tf.keras.Model] = None

    def build_model(self, n_features: int):
        """Build model using the actual number of features from data."""
        self.n_features = n_features
        
        model = models.Sequential([
            layers.Input(shape=(self.seq_length, n_features)),
            layers.LSTM(self.units, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(self.units // 2, return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)   # Predicting next close price
        ])

        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        self.model = model
        logger.info(f"Built LSTM model with {self.seq_length} timesteps and {n_features} features")
        return model

    def prepare_sequences(self, df: pd.DataFrame, target_col: str = 'close') -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences - automatically uses all columns except target."""
        feature_cols = [col for col in df.columns if col != target_col]
        self.n_features = len(feature_cols)   # Update feature count

        data = df[feature_cols + [target_col]].values

        X, y = [], []
        for i in range(len(data) - self.seq_length):
            X.append(data[i:i + self.seq_length])
            y.append(data[i + self.seq_length, -1])   # target value

        return np.array(X), np.array(y)

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 20, batch_size: int = 32, validation_split: float = 0.2):
        if self.model is None:
            # Build model using actual number of features from X
            self.build_model(X.shape[2])

        logger.info(f"Training on {X.shape[0]} samples for {epochs} epochs")
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1,
            shuffle=False
        )
        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained yet.")
        return self.model.predict(X, verbose=0).flatten()

    def save_model(self, filepath: str = "models/lstm_forecaster.keras"):
        if self.model:
            self.model.save(filepath)
            logger.info(f"Model saved to {filepath}")


# Quick test
if __name__ == "__main__":
    print("✅ lstm_forecaster.py (dynamic features version) loaded successfully")