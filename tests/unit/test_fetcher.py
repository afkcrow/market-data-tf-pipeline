import pytest
from src.data.fetcher import FetchConfig, MarketDataFetcher
import pandas as pd

@pytest.mark.asyncio
async def test_fetcher_initialization():
    config = FetchConfig(symbols=["BTC/USDT"], timeframe="15m")
    fetcher = MarketDataFetcher(config)
    assert fetcher.config.exchange_id == "coinbase"
    assert len(fetcher.config.symbols) == 1

@pytest.mark.asyncio
async def test_fetch_ohlcv():
    config = FetchConfig(symbols=["BTC/USDT"], timeframe="15m", limit_per_request=5)
    fetcher = MarketDataFetcher(config)
    df = await fetcher.fetch_ohlcv("BTC/USDT", limit=5)
    assert isinstance(df, pd.DataFrame)