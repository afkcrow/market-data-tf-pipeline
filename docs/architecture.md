# Architecture

## Overview

The pipeline is split into five independent layers — each layer is a single Python module with no circular imports. Data flows strictly left to right; nothing downstream touches the exchange or raw files.

```
Exchange (CCXT)
      │
      ▼
┌─────────────┐     Parquet      ┌──────────────────┐     scaled arrays   ┌────────────────┐
│  data/      │ ────────────▶   │  features/        │ ─────────────────▶ │  models/       │
│  fetcher.py │                  │  technical.py     │                     │  trainer.py    │
│             │                  │                   │                     │  lstm_         │
│  CCXT async │                  │  RSI, VR, SMA,    │                     │  forecaster.py │
│  → Parquet  │                  │  EMA, volatility  │                     │                │
└─────────────┘                  └──────────────────┘                     └────────────────┘
                                                                                   │
                                            ┌──────────────────────────────────────┤
                                            │                                      │
                                            ▼                                      ▼
                                   ┌─────────────────┐                   ┌─────────────────┐
                                   │  inference/     │                   │  evaluation/    │
                                   │  predictor.py   │                   │  backtester.py  │
                                   │                 │                   │                 │
                                   │  model+scaler   │                   │  directional    │
                                   │  +meta → price  │                   │  P&L, Sharpe,   │
                                   └─────────────────┘                   │  drawdown       │
                                                                         └─────────────────┘
                                                                                   │
                                                                                   ▼
                                                                        ┌─────────────────┐
                                                                        │  app.py         │
                                                                        │  Streamlit UI   │
                                                                        └─────────────────┘
```

## Module guide

### `src/config/settings.py`

Pydantic `BaseModel` configuration for the fetcher. All tunable parameters (exchange, symbols, timeframe, output directory) live here. Pass a custom `FetchConfig` instance to override defaults; the rest of the code reads from it.

### `src/data/fetcher.py` — `MarketDataFetcher`

Async CCXT wrapper. Key design points:

- **Public endpoints only** — `enableRateLimit=True`, no API keys.
- **Incremental updates** — on subsequent fetches, reads the latest timestamp from the existing Parquet file and requests only newer candles. New candles are merged with existing history (deduped on index) before saving. This makes re-fetching cheap.
- **Any CCXT exchange** — swap `exchange_id` in `FetchConfig` to use Binance, Bybit, Kraken, etc. with no code changes.
- Output: `data/raw/<SYMBOL>_<timeframe>.parquet` (zstd-compressed).

### `src/features/technical.py`

Pure-pandas feature engineering. Each function takes a DataFrame and returns a DataFrame with new columns appended:

| Function | Columns added |
|---|---|
| `add_rsi` | `rsi` |
| `add_volume_ratio` | `volume_ratio`, `volume_ratio_ma` |
| `add_moving_averages` | `sma_20`, `ema_12`, `ema_26` |
| `add_volatility` | `returns`, `volatility_20` |
| `add_all_features` | all of the above |

No external indicator libraries — zero dependency risk.

### `src/models/base_model.py` — `BaseForecaster`

Abstract base class defining the interface any forecaster must implement: `prepare_sequences`, `train`, `predict`, `save_model`. Swapping the LSTM for a Transformer or N-BEATS model requires implementing this interface and pointing `PipelineTrainer` at the new class.

### `src/models/lstm_forecaster.py` — `LSTMForecaster`

Two-layer LSTM with dropout. Key points:

- **Dynamic feature count** — `build_model(n_features)` is called automatically from `train()` if no model has been built yet, so the architecture adapts to whatever columns `add_all_features` produces.
- **Target column ordering** — `prepare_sequences` always moves `close` to the last column position. `Predictor` mirrors this at inference time so the prediction head always reads from the same index.
- **EarlyStopping** — only enabled when `validation_split > 0` and there are enough samples to split.

### `src/models/trainer.py` — `PipelineTrainer`

Orchestrates the full training run:

1. Load Parquet → add features → drop NaNs.
2. Chronological 80/20 train/test split.
3. Fit `MinMaxScaler` **on training data only** (prevents leakage — the integration suite includes a regression test for this).
4. Build sequences → train LSTM.
5. Inverse-transform predictions back to USD.
6. Serialize: `models/lstm_<SYMBOL>.keras`, `lstm_<SYMBOL>_scaler.pkl`, `lstm_<SYMBOL>_meta.json`.

The metadata JSON records `feature_columns`, `close_col_idx`, and `seq_length` so `Predictor` can reconstruct the exact column order used during training.

### `src/inference/predictor.py` — `Predictor`

Stateless inference: loads `model + scaler + meta` and exposes `predict_next(df)`. Validates that the incoming DataFrame has the expected columns before scaling — catches data-drift mismatches at the boundary. Use `Predictor.for_symbol("BTC-USDT")` as a convenience constructor.

### `src/evaluation/backtester.py` — `Backtester`

Long-only directional strategy: enter when the model predicts price up, exit when it predicts down. Applies a symmetric fee (`fee_bps`) on both entry and exit. Reports:

- Total return
- Sharpe ratio (annualized using `periods_per_year(timeframe)` — crypto-correct, 24/7 calendar)
- Maximum drawdown
- Full equity curve

`periods_per_year` parses CCXT-style timeframe strings (`15m`, `4h`, `1d`, `1w`) so the annualization factor is always correct for the configured candle size.

### `app.py`

Streamlit dashboard with four tabs:

| Tab | Content |
|---|---|
| Market Data | Raw OHLCV table + price chart |
| Technical Features | RSI, Volume Ratio, Volatility, SMA |
| LSTM Forecast | Train model → predictions vs actuals chart + equity curve + Sharpe/drawdown metrics |
| About | Project description |

Training results are stored in `st.session_state` so the charts persist across sidebar interactions without retraining.

## Design decisions

**No data leakage.** The `MinMaxScaler` is fit exclusively on the training split. `test_trainer_no_data_leakage` in the integration suite injects a 10× price spike into the test portion and asserts that training-set values stay within `[0, 1]` while test-set values exceed it — confirming the scaler never saw future data.

**Inference can't drift.** `trainer.py` writes column order and `seq_length` to a sidecar JSON. `Predictor` reads it and rejects DataFrames with wrong columns. Renaming a feature column will raise immediately rather than silently producing garbage predictions.

**Timeframe-aware Sharpe.** Hard-coding 252 (equity trading days) would be wrong for a 15-minute crypto feed. `periods_per_year` converts any CCXT timeframe to the correct annualization factor for a 24/7 market.

**Public data only.** The fetcher uses unauthenticated endpoints. No API keys are read, stored, or required anywhere in the codebase.
