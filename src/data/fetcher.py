"""
src/data/fetcher.py
Modular, public-only market data fetcher using CCXT.
Supports any exchange, multiple symbols, and timeframes.
No API keys required — uses only public endpoints.
"""

from __future__ import annotations

import ccxt
import ccxt.async_support as ccxt_async  
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict
import asyncio
from pydantic import BaseModel, Field
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FetchConfig(BaseModel):
    exchange_id: str = Field(default="coinbase")
    symbols: List[str] = Field(default=["BTC/USDT", "ETH/USDT"])
    timeframe: str = Field(default="15m")
    data_dir: Path = Field(default=Path("data/raw"))
    limit_per_request: int = Field(default=1000)


class MarketDataFetcher:
    def __init__(self, config: Optional[FetchConfig] = None):
        self.config = config or FetchConfig()
        self.config.data_dir.mkdir(parents=True, exist_ok=True)

        # Use async_support for async methods
        self.exchange = getattr(ccxt_async, self.config.exchange_id)({
            'enableRateLimit': True,
        })
        logger.info(f"Initialized public {self.config.exchange_id} client for {len(self.config.symbols)} symbols")

    async def fetch_ohlcv(
        self, symbol: str, since: Optional[int] = None, limit: Optional[int] = None
    ) -> pd.DataFrame:
        try:
            limit = limit or self.config.limit_per_request
            raw_data = await self.exchange.fetch_ohlcv(   # ← changed to fetch_ohlcv (no "async_" prefix)
                symbol=symbol,
                timeframe=self.config.timeframe,
                since=since,
                limit=limit,
            )

            if not raw_data:
                logger.warning(f"No data for {symbol}")
                return pd.DataFrame()

            df = pd.DataFrame(raw_data, columns=["opentime", "open", "high", "low", "close", "volume"])
            df["opentime"] = pd.to_datetime(df["opentime"], unit="ms", utc=True)
            df = df.set_index("opentime").sort_index()

            logger.info(f"Fetched {len(df)} candles for {symbol} ({self.config.timeframe})")
            return df

        except Exception as e:
            logger.error(f"Failed to fetch {symbol}: {e}")
            return pd.DataFrame()

    def save_to_parquet(self, df: pd.DataFrame, symbol: str) -> Optional[Path]:
        if df.empty:
            return None
        safe_symbol = symbol.replace("/", "-")
        file_path = self.config.data_dir / f"{safe_symbol}_{self.config.timeframe}.parquet"
        df.to_parquet(file_path, compression="zstd")
        logger.info(f"Saved to {file_path}")
        return file_path

    async def update_all_symbols(self) -> Dict[str, Optional[Path]]:
        results = {}
        for symbol in self.config.symbols:
            result = await self._update_single_symbol(symbol)
            if result:
                results[symbol] = result
        await self.exchange.close()  # important for async_support
        return results

    async def _update_single_symbol(self, symbol: str) -> Optional[Path]:
        file_path = self.config.data_dir / f"{symbol.replace('/', '-')}_{self.config.timeframe}.parquet"

        since_ms: Optional[int] = None
        if file_path.exists():
            try:
                existing = pd.read_parquet(file_path)
                if not existing.empty:
                    last_ts = existing.index.max()
                    since_ms = int(last_ts.timestamp() * 1000) + 1
            except Exception:
                pass

        df = await self.fetch_ohlcv(symbol, since=since_ms)
        if not df.empty:
            return self.save_to_parquet(df, symbol)
        return None


if __name__ == "__main__":
    async def main():
        config = FetchConfig(timeframe="15m")
        fetcher = MarketDataFetcher(config)
        await fetcher.update_all_symbols()
        print("✅ Fetch complete — check data/raw/ folder")

    asyncio.run(main())