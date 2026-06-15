"""
tests/integration/test_pipeline.py
Integration tests: feature engineering → scaling → sequences → model → backtester.
Uses synthetic data so no exchange connection is required.
"""

import numpy as np
import pandas as pd

from src.evaluation.backtester import Backtester
from src.features.technical import add_all_features
from src.models.lstm_forecaster import LSTMForecaster
from src.models.trainer import PipelineTrainer


def _make_ohlcv(n: int = 300) -> pd.DataFrame:
    np.random.seed(42)
    close = 30_000 + np.cumsum(np.random.randn(n) * 50)
    return pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.002,
            "low": close * 0.998,
            "close": close,
            "volume": np.random.randint(1_000, 5_000, n).astype(float),
        }
    )


def test_features_produce_correct_columns():
    df = add_all_features(_make_ohlcv()).dropna()
    expected = {
        "rsi",
        "sma_20",
        "ema_12",
        "ema_26",
        "volatility_20",
        "volume_ratio",
        "volume_ratio_ma",
    }
    assert expected.issubset(set(df.columns))


def test_sequences_shape():
    df = add_all_features(_make_ohlcv()).dropna()
    forecaster = LSTMForecaster(seq_length=20)
    X, y = forecaster.prepare_sequences(df)

    assert X.ndim == 3
    assert X.shape[1] == 20
    assert X.shape[0] == y.shape[0]
    assert X.shape[0] == len(df) - 20


def test_trainer_scales_data():
    """Scaler should map all features to [0, 1] on training split."""
    trainer = PipelineTrainer()
    df = add_all_features(_make_ohlcv(400)).dropna()

    X_train, X_test, y_train, y_test = trainer.prepare_train_test_split(df)

    assert trainer.scaler is not None
    # Training sequences should be in [0, 1]
    assert X_train.min() >= -1e-6
    assert X_train.max() <= 1.0 + 1e-6


def test_trainer_no_data_leakage():
    """Scaler fitted only on train — test set may exceed [0,1] if prices diverge."""
    trainer = PipelineTrainer()
    df = add_all_features(_make_ohlcv(400)).dropna()
    # Inject a large spike in the test portion to confirm scaler saw only train
    df_modified = df.copy()
    df_modified.iloc[int(len(df) * 0.85) :, df.columns.get_loc("close")] *= 10

    X_train, X_test, _, _ = trainer.prepare_train_test_split(df_modified)
    # Train should still be in [0, 1]; test should exceed 1 due to unseen spike
    assert X_train.max() <= 1.0 + 1e-6
    assert X_test.max() > 1.0


def test_model_train_and_predict():
    trainer = PipelineTrainer()
    df = add_all_features(_make_ohlcv(400)).dropna()

    X_train, X_test, y_train, y_test = trainer.prepare_train_test_split(df)
    trainer.forecaster.train(X_train, y_train, epochs=1, batch_size=32)

    preds = trainer.forecaster.predict(X_test)
    assert preds.shape == y_test.shape


def test_inverse_transform_returns_price_scale():
    trainer = PipelineTrainer()
    df = add_all_features(_make_ohlcv(400)).dropna()

    _, _, _, y_test = trainer.prepare_train_test_split(df)
    prices = trainer.inverse_transform_prices(y_test)

    # Should be back in the original BTC-like price range, not [0, 1]
    assert prices.mean() > 100


def test_backtester_basic():
    actual = np.array([100.0, 102.0, 101.0, 105.0, 103.0, 107.0, 110.0])
    predicted = np.array([101.0, 100.0, 104.0, 102.0, 106.0, 109.0, 108.0])

    bt = Backtester(initial_capital=1_000.0, fee_bps=0.0)
    results = bt.run(actual, predicted)

    assert "total_return" in results
    assert "sharpe_ratio" in results
    assert results["max_drawdown"] >= 0.0
    assert len(results["equity_curve"]) == len(actual)


def test_backtester_buy_and_hold_beats_flat():
    """A perfect predictor should outperform staying flat."""
    n = 50
    actual = np.linspace(100, 200, n)  # steadily rising
    predicted = np.roll(actual, -1)  # always predicts next correctly
    predicted[-1] = predicted[-2]

    bt = Backtester(initial_capital=1_000.0, fee_bps=0.0)
    results = bt.run(actual, predicted)
    assert results["total_return"] > 0
