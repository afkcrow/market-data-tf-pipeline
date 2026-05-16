from pydantic import BaseModel
from pathlib import Path

class FetchConfig(BaseModel):
    # (same class as in fetcher.py — you can import from here later)
    exchange_id: str = Field(..., description="CCXT exchange id (e.g. binance, bybit, coinbase)")
    symbols: list[str] = ["BTC/USDT", "ETH/USDT"]
    timeframe: str = "15m"
    data_dir: Path = Path("data/raw")