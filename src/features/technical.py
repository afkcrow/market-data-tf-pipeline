"""
src/features/technical.py
Standard technical indicators using pure pandas (no extra dependencies).
All features are generic and public — zero alpha or IP risk.
"""

import pandas as pd


def add_volume_ratio(df: pd.DataFrame, period: int = 14, ma_period: int = 6) -> pd.DataFrame:
    """Volume Ratio (VR) - measures buying vs selling pressure."""
    df = df.copy()
    change = df["close"].diff()
    up_vol = df["volume"].where(change > 0, 0)
    down_vol = df["volume"].where(change < 0, 0)
    flat_vol = df["volume"].where(change == 0, 0)

    th = up_vol.rolling(period).sum()
    tl = down_vol.rolling(period).sum()
    tq = flat_vol.rolling(period).sum()

    df["volume_ratio"] = 100 * (th * 2 + tq) / (tl * 2 + tq + 1e-9)
    df["volume_ratio_ma"] = df["volume_ratio"].ewm(span=ma_period).mean()
    return df


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Relative Strength Index (standard 14-period)."""
    df = df.copy()
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss.replace(0, 1e-9)
    df["rsi"] = 100 - (100 / (1 + rs))
    return df


def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """SMA and EMA (common periods)."""
    df = df.copy()
    df["sma_20"] = df["close"].rolling(20).mean()
    df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["ema_26"] = df["close"].ewm(span=26, adjust=False).mean()
    return df


def add_volatility(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Rolling volatility (standard deviation of returns)."""
    df = df.copy()
    df["returns"] = df["close"].pct_change()
    df["volatility_20"] = df["returns"].rolling(period).std()
    return df


def add_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all standard features in one call."""
    df = add_volume_ratio(df)
    df = add_rsi(df)
    df = add_moving_averages(df)
    df = add_volatility(df)
    return df


# Quick test (run with: uv run -m src.features.technical)
if __name__ == "__main__":
    # Example usage
    print("✅ technical.py loaded successfully")
    print("Use: df = add_all_features(your_dataframe)")
