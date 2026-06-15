"""
scripts/predict.py
CLI wrapper: load a trained model + scaler + metadata and predict the next close price.

Usage:
    uv run python scripts/predict.py --symbol BTC-USDT
    uv run python scripts/predict.py --symbol ETH-USDT --timeframe 4h
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Make `src` importable when the script is run directly (not via -m).
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.inference.predictor import Predictor


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict the next close price for a trained symbol."
    )
    parser.add_argument(
        "--symbol",
        default="BTC-USDT",
        help="Symbol slug matching the trained model, e.g. BTC-USDT (default: BTC-USDT)",
    )
    parser.add_argument(
        "--timeframe",
        default="15m",
        help="Candle timeframe, e.g. 15m, 4h, 1d (default: 15m)",
    )
    parser.add_argument(
        "--data-dir",
        default="data/raw",
        help="Directory containing Parquet data files (default: data/raw)",
    )
    parser.add_argument(
        "--models-dir",
        default="models",
        help="Directory containing model artifacts (default: models)",
    )
    args = parser.parse_args()

    # Validate user input before using it to build filesystem paths (prevents traversal).
    if not re.fullmatch(r"[A-Za-z0-9-]+", args.symbol):
        print(f"Error: invalid symbol {args.symbol!r} (allowed: letters, digits, '-')")
        raise SystemExit(1)
    if not re.fullmatch(r"\d+[mhdw]", args.timeframe):
        print(f"Error: invalid timeframe {args.timeframe!r} (e.g. 15m, 4h, 1d)")
        raise SystemExit(1)

    data_path = Path(args.data_dir) / f"{args.symbol}_{args.timeframe}.parquet"
    if not data_path.exists():
        print(f"Error: data file not found at {data_path}")
        print("Run 'make fetch' (or 'uv run python -m src.data.fetcher') first.")
        raise SystemExit(1)

    model_path = Path(args.models_dir) / f"lstm_{args.symbol}.keras"
    if not model_path.exists():
        print(f"Error: trained model not found at {model_path}")
        print("Run 'make train' (or 'uv run python -m src.models.trainer') first.")
        raise SystemExit(1)

    df = pd.read_parquet(data_path)
    predictor = Predictor.for_symbol(symbol=args.symbol, models_dir=args.models_dir)
    price = predictor.predict_next(df)

    print(f"\nSymbol   : {args.symbol}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Latest close  : ${df['close'].iloc[-1]:,.2f}")
    print(f"Predicted next: ${price:,.2f}")
    delta_pct = (price - df["close"].iloc[-1]) / df["close"].iloc[-1] * 100
    direction = "UP" if delta_pct > 0 else "DOWN"
    print(f"Signal   : {direction} ({delta_pct:+.2f}%)\n")


if __name__ == "__main__":
    main()
