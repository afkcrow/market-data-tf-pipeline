# Contributing

Thanks for your interest. This is a portfolio project but PRs and issues are welcome.

## Setup

```bash
git clone https://github.com/afkcrow/market-data-tf-pipeline.git
cd market-data-tf-pipeline
uv sync
```

## Development workflow

```bash
make fetch       # pull latest candle data
make train       # train LSTM on BTC-USDT
make test        # run full test suite
make lint        # ruff check
make fmt         # ruff format (auto-fix)
make typecheck   # mypy
make app         # launch Streamlit dashboard
```

All of `lint`, `fmt-check`, `typecheck`, and `test` run in CI on every push. Your PR must pass all four.

## Code style

- **Formatter:** ruff (line length 100). Run `make fmt` before committing.
- **Linter:** ruff with `E, F, I, UP, W` rules. No `# noqa` suppressions without a comment explaining why.
- **Types:** Python 3.12 native types (`X | None`, `list[str]`, `tuple[int, ...]`). No `from typing import Optional/Tuple/Dict`.
- **Comments:** only for non-obvious invariants or workarounds. No docstring novels.

## Adding a new feature indicator

1. Add a function to `src/features/technical.py` following the `add_*` pattern (takes a DataFrame, returns a DataFrame with new column(s)).
2. Call it from `add_all_features`.
3. Add a unit test in `tests/unit/test_features.py`.

## Adding a new model

1. Subclass `BaseForecaster` from `src/models/base_model.py`.
2. Implement `prepare_sequences`, `train`, `predict`, `save_model`.
3. Point `PipelineTrainer.forecaster` at your class.

## Running tests

```bash
make test          # full suite, verbose
make test-fast     # stop on first failure, quiet
```

Tests use synthetic data and a mocked exchange — no network access required.

## Submitting a PR

- Keep PRs focused: one feature or fix per PR.
- Include a brief description of what changed and why.
- All CI checks must pass before review.
