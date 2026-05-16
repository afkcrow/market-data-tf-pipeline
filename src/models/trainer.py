"""
src/models/trainer.py
Updated to work with the new dynamic LSTMForecaster.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict
import logging

from src.features.technical import add_all_features
from src.models.lstm_forecaster import LSTMForecaster

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PipelineTrainer:
    """Main training pipeline."""

    def __init__(self, data_dir: Path = Path("data/raw"), timeframe: str = "15m"):
        self.data_dir = data_dir
        self.timeframe = timeframe
        self.forecaster = LSTMForecaster(seq_length=60)   # No n_features here anymore

    def load_and_prepare_data(self, symbol: str = "BTC-USDT") -> pd.DataFrame:
        """Load data and apply all standard features."""
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
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare sequences for training."""
        X, y = self.forecaster.prepare_sequences(df)

        split_idx = int(len(X) * train_ratio)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        return X_train, X_test, y_train, y_test

    def train_model(
        self,
        symbol: str = "BTC-USDT",
        epochs: int = 10,      # Small number for fast testing
        batch_size: int = 32
    ) -> Dict:
        """End-to-end training pipeline."""
        logger.info(f"Starting training for {symbol}")

        # 1. Load + engineer features
        df = self.load_and_prepare_data(symbol)

        # 2. Prepare sequences
        X_train, X_test, y_train, y_test = self.prepare_train_test_split(df)

        # 3. Train model (now dynamic)
        history = self.forecaster.train(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2
        )

        # 4. Quick test evaluation
        test_predictions = self.forecaster.predict(X_test)
        test_mae = float(np.mean(np.abs(test_predictions - y_test)))

        logger.info(f"Training finished. Test MAE: {test_mae:.4f}")

        # Save model
        model_path = f"models/lstm_{symbol.replace('/', '-')}.keras"
        self.forecaster.save_model(model_path)

        return {
            "test_mae": test_mae,
            "model_path": model_path,
            "train_history": history.history if history else None
        }


# Quick test
if __name__ == "__main__":
    trainer = PipelineTrainer()
    try:
        results = trainer.train_model(epochs=8)   # Small epochs for quick test
        print("\n✅ Training completed successfully!")
        print(f"Test MAE: {results['test_mae']:.4f}")
        print(f"Model saved to: {results['model_path']}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Tip: Make sure you ran the fetcher first (`uv run -m src.data.fetcher`)")