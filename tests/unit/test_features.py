import pandas as pd

from src.features.technical import add_all_features, add_rsi, add_volume_ratio


def test_volume_ratio():
    df = pd.DataFrame({"close": [100, 102, 101, 105, 103], "volume": [1000, 1200, 800, 1500, 900]})
    result = add_volume_ratio(df, period=3)
    assert "volume_ratio" in result.columns
    assert result["volume_ratio"].notna().any()


def test_rsi():
    df = pd.DataFrame({"close": range(20)})
    result = add_rsi(df, period=14)
    assert "rsi" in result.columns


def test_add_all_features():
    df = pd.DataFrame({"close": range(50), "volume": [1000 + i * 10 for i in range(50)]})
    result = add_all_features(df)
    assert len(result.columns) > 5
    assert not result.isnull().values.all()
