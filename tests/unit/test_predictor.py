"""Unit tests for the inference Predictor's anti-drift guards.

The model is stubbed so these stay fast and offline; the point is to exercise
the metadata/column checks that keep training and serving in sync.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import MinMaxScaler

from src.features.technical import add_all_features
from src.inference.predictor import Predictor


class _StubModel:
    """Stand-in for a Keras model so the test never touches TensorFlow training."""

    def predict(self, X: np.ndarray, verbose: int = 0) -> np.ndarray:
        return np.array([[0.42]])


def _make_raw_ohlcv(n: int = 150) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    volume = rng.uniform(800, 1500, n)
    return pd.DataFrame({"close": close, "volume": volume})


def _fit_scaler() -> tuple[list[str], MinMaxScaler]:
    engineered = add_all_features(_make_raw_ohlcv()).dropna()
    cols = list(engineered.columns)
    scaler = MinMaxScaler().fit(engineered.values)
    return cols, scaler


def _write_artifacts(
    tmp_path, feature_columns: list[str], close_col_idx: int, scaler: MinMaxScaler
) -> tuple[str, str, str]:
    model_path = tmp_path / "lstm_TEST.keras"
    scaler_path = tmp_path / "lstm_TEST_scaler.pkl"
    meta_path = tmp_path / "lstm_TEST_meta.json"
    model_path.write_text("stub")  # never read — load_model is patched
    joblib.dump(scaler, scaler_path)
    meta_path.write_text(
        json.dumps(
            {
                "symbol": "TEST",
                "timeframe": "15m",
                "seq_length": 10,
                "feature_columns": feature_columns,
                "close_col_idx": close_col_idx,
            }
        )
    )
    return str(model_path), str(scaler_path), str(meta_path)


@patch("src.inference.predictor.tf.keras.models.load_model", return_value=_StubModel())
def test_predict_next_returns_price_when_schema_matches(_mock_load, tmp_path):
    cols, scaler = _fit_scaler()
    paths = _write_artifacts(tmp_path, cols, cols.index("close"), scaler)
    predictor = Predictor(*paths)

    result = predictor.predict_next(_make_raw_ohlcv())

    assert isinstance(result, float)


@patch("src.inference.predictor.tf.keras.models.load_model", return_value=_StubModel())
def test_predict_next_rejects_drifted_schema(_mock_load, tmp_path):
    cols, scaler = _fit_scaler()
    # Metadata expects a feature the current pipeline no longer produces.
    drifted = cols + ["__removed_feature__"]
    paths = _write_artifacts(tmp_path, drifted, cols.index("close"), scaler)
    predictor = Predictor(*paths)

    with pytest.raises(ValueError, match="missing columns"):
        predictor.predict_next(_make_raw_ohlcv())
