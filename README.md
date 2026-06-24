# Market Data Pipeline + TensorFlow Forecasting

**A modular, production-minded end-to-end market data pipeline with feature engineering, LSTM forecasting, backtesting, and an interactive Streamlit dashboard.**

Built as a portfolio project to demonstrate clean engineering practices in financial time-series ML — no proprietary alpha, just a transparent reference implementation.

[![CI](https://github.com/afkcrow/market-data-tf-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/afkcrow/market-data-tf-pipeline/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

---

## Highlights

- **Modular data fetching** with CCXT — switch between 100+ exchanges via config, no API keys required for public OHLCV.
- **Pure-pandas technical features** — RSI, volume ratio, SMAs/EMAs, rolling volatility. Zero non-trivial dependencies.
- **TensorFlow / Keras LSTM** that infers feature count dynamically and respects the train/test boundary (scaler fitted on train only).
- **Inference path** that round-trips through saved `model + scaler + metadata`, so column order and `seq_length` can't drift between training and serving.
- **Directional backtester** with fees, equity curve, Sharpe (annualized to the configured timeframe), and max drawdown.
- **Streamlit dashboard** to fetch, train, and visualize predictions side-by-side with actuals.
- **Production-shaped repo**: `pyproject.toml` + `uv`, ruff + mypy, GitHub Actions CI, multi-stage Dockerfile, tests with mocked exchange.

## Tech stack

Python 3.12 · CCXT · TensorFlow / Keras · pandas · scikit-learn · Pydantic · Streamlit · Plotly · uv · ruff · pytest

---

## Quick start

### Install

```bash
git clone https://github.com/afkcrow/market-data-tf-pipeline.git
cd market-data-tf-pipeline
uv sync
```

### Fetch data

```bash
uv run python -m src.data.fetcher
# or
make fetch
```

This pulls 15-minute OHLCV for `BTC/USDT` and `ETH/USDT` from Coinbase (default) and writes Parquet files to `data/raw/`. Edit `src/config/settings.py` or pass a `FetchConfig` to use a different exchange, symbol list, or timeframe.

### Train

```bash
uv run python -m src.models.trainer
# or
make train
```

Trains the LSTM, saves `models/lstm_<SYMBOL>.keras`, the fitted `MinMaxScaler`, and a `*_meta.json` describing column order and sequence length.

### Run the dashboard

```bash
uv run streamlit run app.py
# or
make app
```

Open <http://localhost:8501>.

### Predict from the CLI

```bash
uv run python scripts/predict.py --symbol BTC-USDT
```

---

## Project layout

```
market-data-tf-pipeline/
├── app.py                          # Streamlit dashboard
├── src/
│   ├── config/settings.py          # Pydantic config models
│   ├── data/fetcher.py             # Async CCXT OHLCV fetcher → Parquet
│   ├── features/technical.py       # RSI, volume ratio, MAs, volatility
│   ├── models/
│   │   ├── base_model.py           # BaseForecaster ABC
│   │   ├── lstm_forecaster.py      # Keras LSTM that adapts to feature count
│   │   └── trainer.py              # End-to-end training + serialization
│   ├── inference/predictor.py      # Load model+scaler+meta, predict next close
│   └── evaluation/backtester.py    # Directional backtest with fees + Sharpe
├── tests/
│   ├── unit/                       # Mocked-exchange + feature tests
│   └── integration/                # Pipeline + backtester end-to-end
├── scripts/                        # CLI entry points (train / predict)
├── deployment/Dockerfile           # Multi-stage Streamlit container
├── docs/architecture.md            # System diagram + module guide
└── .github/workflows/              # CI: lint, type-check, test, fmt-check
```

## Architecture

```
            ┌──────────────┐    ┌──────────────────┐    ┌─────────────────┐
            │  CCXT (any   │    │  Feature engine  │    │  LSTM forecaster│
  fetcher ─▶│  exchange)   │─▶─▶│  RSI, VR, SMA,   │─▶─▶│  scaler + meta  │
            │  → Parquet   │    │  EMA, volatility │    │  → .keras       │
            └──────────────┘    └──────────────────┘    └─────────────────┘
                                                                │
                                                                ▼
                                                     ┌────────────────────┐
                                                     │  Predictor         │
                                                     │  + Backtester      │
                                                     │  + Streamlit UI    │
                                                     └────────────────────┘
```

See [docs/architecture.md](docs/architecture.md) for the full module-by-module breakdown.

## Development

```bash
make sync          # uv sync
make test          # pytest (unit + integration, mocked exchange)
make lint          # ruff check
make fmt           # ruff format
make fmt-check     # ruff format --check
make typecheck     # mypy src/
make clean         # remove caches
```

CI runs the equivalent on every push and pull request.

### Docker

```bash
docker build -f deployment/Dockerfile -t market-data-tf-pipeline .
docker run -p 8501:8501 market-data-tf-pipeline
```

## Design notes

- **Honest baseline.** Training reports the LSTM against a naive persistence forecast (next close = current close) via a `skill_score` and directional accuracy (`src/evaluation/baseline.py`). On noisy 24/7 crypto a next-step model that doesn't clear persistence hasn't learned much — this makes that explicit instead of hiding it.
- **No data leakage.** `MinMaxScaler` is fit on the training split only; the test set is transformed with the same scaler and the integration suite includes an explicit regression test (`test_trainer_no_data_leakage`).
- **Inference can't drift.** `trainer.py` writes a `*_meta.json` with `feature_columns`, `close_col_idx`, and `seq_length`; `Predictor` reads it and rejects DataFrames with the wrong columns.
- **Timeframe-aware Sharpe.** `periods_per_year('15m')` converts the configured CCXT timeframe into the right annualization factor — no hard-coded 252.
- **Public data only.** The fetcher uses unauthenticated CCXT endpoints; no API keys are read or required.

## Roadmap

- [ ] Transformer / N-BEATS baseline for comparison
- [ ] Walk-forward cross-validation in the backtester
- [ ] Per-feature importance via SHAP
- [ ] MLflow experiment tracking

## License

[MIT](LICENSE) — see `LICENSE` for details.
