"""
src/models/lstm_forecaster.py
LSTM model that adapts to the number of features in the input data.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models

from src.models.base_model import BaseForecaster

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LSTMForecaster(BaseForecaster):
    """LSTM forecaster that infers the feature count from the input tensor."""

    def __init__(self, seq_length: int = 60, units: int = 64):
        self.seq_length = seq_length
        self.units = units
        self.n_features: int | None = None
        self.model: tf.keras.Model | None = None

    def build_model(self, n_features: int) -> tf.keras.Model:
        self.n_features = n_features

        model = models.Sequential(
            [
                layers.Input(shape=(self.seq_length, n_features)),
                layers.LSTM(self.units, return_sequences=True),
                layers.Dropout(0.2),
                layers.LSTM(self.units // 2, return_sequences=False),
                layers.Dropout(0.2),
                layers.Dense(32, activation="relu"),
                layers.Dense(1),
            ]
        )

        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        self.model = model
        logger.info(f"Built LSTM model with {self.seq_length} timesteps and {n_features} features")
        return model

    def prepare_sequences(
        self, df: pd.DataFrame, target_col: str = "close"
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build (X, y). Target is placed last so y is always the final column."""
        feature_cols = [c for c in df.columns if c != target_col]
        self.n_features = len(feature_cols)

        data = df[feature_cols + [target_col]].values

        n_windows = len(data) - self.seq_length
        if n_windows <= 0:
            return np.empty((0, self.seq_length, data.shape[1])), np.empty((0,))

        X = np.empty((n_windows, self.seq_length, data.shape[1]), dtype=data.dtype)
        y = np.empty(n_windows, dtype=data.dtype)
        for i in range(n_windows):
            X[i] = data[i : i + self.seq_length]
            y[i] = data[i + self.seq_length, -1]
        return X, y

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 20,
        batch_size: int = 32,
        validation_split: float = 0.2,
        patience: int = 5,
        **kwargs: Any,
    ):
        if self.model is None:
            self.build_model(X.shape[2])
        assert self.model is not None  # build_model always assigns it

        logger.info(f"Training on {X.shape[0]} samples for up to {epochs} epochs")

        # Only enable EarlyStopping when there's a real validation split to monitor.
        cb = []
        if validation_split and validation_split > 0 and len(X) > 1:
            cb.append(
                callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=patience,
                    restore_best_weights=True,
                )
            )

        return self.model.fit(
            X,
            y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1,
            shuffle=False,
            callbacks=cb,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained yet.")
        return np.asarray(self.model.predict(X, verbose=0)).flatten()

    def save_model(self, filepath: str = "models/lstm_forecaster.keras") -> None:
        if self.model:
            self.model.save(filepath)
            logger.info(f"Model saved to {filepath}")


if __name__ == "__main__":
    print("lstm_forecaster.py loaded successfully")
