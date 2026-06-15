"""
src/inference/predictor.py
Load a trained model + scaler + metadata and run inference on the latest market data.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from src.features.technical import add_all_features

logger = logging.getLogger(__name__)


class Predictor:
    def __init__(
        self,
        model_path: str,
        scaler_path: str,
        meta_path: str | None = None,
        seq_length: int | None = None,
    ):
        self.model = tf.keras.models.load_model(model_path)
        self.scaler = joblib.load(scaler_path)

        if meta_path and Path(meta_path).exists():
            meta = json.loads(Path(meta_path).read_text())
            self.feature_columns: list[str] = list(meta["feature_columns"])
            self.close_col_idx: int = int(meta["close_col_idx"])
            self.seq_length: int = int(meta.get("seq_length", seq_length or 60))
        else:
            if seq_length is None:
                raise ValueError(
                    "Metadata not found and seq_length not provided; "
                    "cannot determine input window length."
                )
            self.feature_columns = []
            self.close_col_idx = -1
            self.seq_length = seq_length

        logger.info(f"Loaded model from {model_path}")

    def predict_next(self, df: pd.DataFrame) -> float:
        """Return the predicted next close price given a raw OHLCV DataFrame."""
        df = add_all_features(df).dropna()

        if len(df) < self.seq_length:
            raise ValueError(f"Need at least {self.seq_length} rows, got {len(df)}")

        if self.feature_columns:
            missing = set(self.feature_columns) - set(df.columns)
            if missing:
                raise ValueError(f"Inference DataFrame is missing columns: {sorted(missing)}")
            df = df[self.feature_columns]
            close_col_idx = self.close_col_idx
        else:
            close_col_idx = list(df.columns).index("close")

        if df.shape[1] != self.scaler.n_features_in_:
            raise ValueError(
                f"Feature count mismatch: scaler expects {self.scaler.n_features_in_}, "
                f"got {df.shape[1]}"
            )

        scaled = self.scaler.transform(df.values)

        # Mirror prepare_sequences: move target column to the last position.
        feature_idx = [i for i in range(scaled.shape[1]) if i != close_col_idx]
        ordered = scaled[:, feature_idx + [close_col_idx]]

        sequence = ordered[-self.seq_length :]
        X = sequence[np.newaxis, ...]

        pred_scaled = float(self.model.predict(X, verbose=0).flatten()[0])

        dummy = np.zeros((1, self.scaler.n_features_in_))
        dummy[0, close_col_idx] = pred_scaled
        pred_price = float(self.scaler.inverse_transform(dummy)[0, close_col_idx])

        logger.info(f"Predicted next close: {pred_price:.4f}")
        return pred_price

    @classmethod
    def for_symbol(
        cls,
        symbol: str = "BTC-USDT",
        models_dir: str = "models",
    ) -> Predictor:
        """Convenience constructor: loads the model, scaler, and metadata for a symbol."""
        base = Path(models_dir)
        return cls(
            model_path=str(base / f"lstm_{symbol}.keras"),
            scaler_path=str(base / f"lstm_{symbol}_scaler.pkl"),
            meta_path=str(base / f"lstm_{symbol}_meta.json"),
        )
