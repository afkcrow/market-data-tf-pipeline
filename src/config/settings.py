from pathlib import Path

from pydantic import BaseModel, Field


class FetchConfig(BaseModel):
    exchange_id: str = Field(
        default="coinbase",
        description="CCXT exchange id (e.g. binance, bybit, coinbase)",
    )
    symbols: list[str] = Field(default_factory=lambda: ["BTC/USDT", "ETH/USDT"])
    timeframe: str = Field(default="15m")
    data_dir: Path = Field(default=Path("data/raw"))
    limit_per_request: int = Field(default=1000)
