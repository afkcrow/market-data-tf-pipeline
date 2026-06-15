"""
tests/unit/test_fetcher.py
Unit tests for the fetcher. The exchange call is mocked so CI doesn't need network.
"""

from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from src.config.settings import FetchConfig
from src.data.fetcher import MarketDataFetcher


@pytest.mark.asyncio
async def test_fetcher_initialization():
    config = FetchConfig(symbols=["BTC/USDT"], timeframe="15m")
    fetcher = MarketDataFetcher(config)
    assert fetcher.config.exchange_id == "coinbase"
    assert len(fetcher.config.symbols) == 1
    await fetcher.exchange.close()


@pytest.mark.asyncio
async def test_fetch_ohlcv_parses_candles():
    """fetch_ohlcv should turn raw CCXT candles into a tz-aware indexed DataFrame."""
    config = FetchConfig(symbols=["BTC/USDT"], timeframe="15m")
    fetcher = MarketDataFetcher(config)

    raw_candles = [
        # [timestamp_ms, open, high, low, close, volume]
        [1_700_000_000_000, 100.0, 110.0, 95.0, 105.0, 12.5],
        [1_700_000_900_000, 105.0, 115.0, 100.0, 112.0, 18.0],
    ]

    with patch.object(fetcher.exchange, "fetch_ohlcv", new=AsyncMock(return_value=raw_candles)):
        df = await fetcher.fetch_ohlcv("BTC/USDT", limit=2)

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) == 2
    assert df.index.tz is not None
    assert df["close"].iloc[-1] == 112.0

    await fetcher.exchange.close()


@pytest.mark.asyncio
async def test_fetch_ohlcv_handles_empty():
    """Empty exchange response should return an empty DataFrame, not raise."""
    fetcher = MarketDataFetcher(FetchConfig(symbols=["BTC/USDT"]))

    with patch.object(fetcher.exchange, "fetch_ohlcv", new=AsyncMock(return_value=[])):
        df = await fetcher.fetch_ohlcv("BTC/USDT")

    assert df.empty
    await fetcher.exchange.close()
