"""
src/models/trainer.py
End-to-end training pipeline with per-feature MinMax scaling.
Scaler is fit on the training split only to prevent data leakage.
Saves model + scaler + metadata (column order, seq_length) for inference.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.features.technical import add_all_features
from src.models.lstm_forecaster import LSTMForecaster

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PipelineTrainer:
    def __init__(self, data_dir: Path = Path("data/raw"), timeframe: str = "15m"):
        self.data_dir = data_dir
        self.timeframe = timeframe
        self.forecaster = LSTMForecaster(seq_length=60)
        self.scaler: MinMaxScaler | None = None
        self.feature_columns: list[str] | None = None
        self._close_col_idx: int | None = None

    def load_and_prepare_data(self, symbol: str = "BTC-USDT") -> pd.DataFrame:
        file_path = self.data_dir / f"{symbol}_{self.timeframe}.parquet"

        if not file_path.exists():
            raise FileNotFoundError(f"No data file found: {file_path}. Run fetcher first.")

        logger.info(f"Loading data from {file_path}")
        df = pd.read_parquet(file_path)
        df = add_all_features(df)
        df = df.dropna().copy()

        logger.info(f"Prepared dataset with shape {df.shape} ({len(df.columns)} features)")
        return df

    def prepare_train_test_split(
        self, df: pd.DataFrame, train_ratio: float = 0.8
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        split_idx = int(len(df) * train_ratio)
        df_train = df.iloc[:split_idx]
        df_test = df.iloc[split_idx:]

        # Fit scaler on training data only — prevents leakage of future statistics.
        self.scaler = MinMaxScaler()
        train_scaled = self.scaler.fit_transform(df_train.values)
        test_scaled = self.scaler.transform(df_test.values)

        df_train_scaled = pd.DataFrame(train_scaled, columns=df.columns, index=df_train.index)
        df_test_scaled = pd.DataFrame(test_scaled, columns=df.columns, index=df_test.index)

        # Remember the original column layout so inference can reproduce it exactly.
        self.feature_columns = list(df.columns)
        self._close_col_idx = self.feature_columns.index("close")

        X_train, y_train = self.forecaster.prepare_sequences(df_train_scaled)
        X_test, y_test = self.forecaster.prepare_sequences(df_test_scaled)

        logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
        return X_train, X_test, y_train, y_test

    def inverse_transform_prices(self, scaled_values: np.ndarray) -> np.ndarray:
        """Convert scaled close-price values back to real price units."""
        if self.scaler is None or self._close_col_idx is None:
            raise RuntimeError("Scaler not fitted; call prepare_train_test_split first.")
        n_features = self.scaler.n_features_in_
        dummy = np.zeros((len(scaled_values), n_features))
        dummy[:, self._close_col_idx] = scaled_values
        return self.scaler.inverse_transform(dummy)[:, self._close_col_idx]

    def train_model(
        self,
        symbol: str = "BTC-USDT",
        epochs: int = 10,
        batch_size: int = 32,
        models_dir: Path = Path("models"),
    ) -> dict:
        logger.info(f"Starting training for {symbol}")
        models_dir.mkdir(parents=True, exist_ok=True)

        df = self.load_and_prepare_data(symbol)
        X_train, X_test, y_train, y_test = self.prepare_train_test_split(df)

        history = self.forecaster.train(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
        )

        test_predictions_scaled = self.forecaster.predict(X_test)
        test_predictions = self.inverse_transform_prices(test_predictions_scaled)
        y_test_real = self.inverse_transform_prices(y_test)
        test_mae = float(np.mean(np.abs(test_predictions - y_test_real)))

        logger.info(f"Training finished. Test MAE: ${test_mae:.2f}")

        model_path = models_dir / f"lstm_{symbol}.keras"
        scaler_path = models_dir / f"lstm_{symbol}_scaler.pkl"
        meta_path = models_dir / f"lstm_{symbol}_meta.json"

        self.forecaster.save_model(str(model_path))
        joblib.dump(self.scaler, scaler_path)
        meta_path.write_text(
            json.dumps(
                {
                    "symbol": symbol,
                    "timeframe": self.timeframe,
                    "seq_length": self.forecaster.seq_length,
                    "feature_columns": self.feature_columns,
                    "close_col_idx": self._close_col_idx,
                },
                indent=2,
            )
        )

        logger.info(f"Scaler saved to {scaler_path}")
        logger.info(f"Metadata saved to {meta_path}")

        return {
            "test_mae": test_mae,
            "model_path": str(model_path),
            "scaler_path": str(scaler_path),
            "meta_path": str(meta_path),
            "train_history": history.history if history else None,
            "test_predictions": test_predictions.tolist(),
            "test_actual": y_test_real.tolist(),
        }


if __name__ == "__main__":
    trainer = PipelineTrainer()
    try:
        results = trainer.train_model(epochs=8)
        print("\nTraining completed successfully!")
        print(f"Test MAE: ${results['test_mae']:.2f}")
        print(f"Model saved to: {results['model_path']}")
    except Exception as e:
        print(f"\nError: {e}")
        print("Tip: Run the fetcher first (`uv run -m src.data.fetcher`)")
