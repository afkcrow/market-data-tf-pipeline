"""
src/models/base_model.py
Minimal forecaster interface so multiple model families can plug into the trainer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseForecaster(ABC):
    """Contract every forecaster implementation must satisfy."""

    seq_length: int

    @abstractmethod
    def prepare_sequences(
        self, df: pd.DataFrame, target_col: str = "close"
    ) -> tuple[np.ndarray, np.ndarray]: ...

    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs): ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def save_model(self, filepath: str) -> None: ...
