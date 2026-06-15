.PHONY: sync test test-fast lint fmt fmt-check typecheck fetch train predict app clean

sync:
	uv sync

test:
	uv run pytest tests/ -v --tb=short

test-fast:
	uv run pytest tests/ -x --tb=short -q

lint:
	uv run ruff check .

fmt:
	uv run ruff format .

fmt-check:
	uv run ruff format --check .

typecheck:
	uv run mypy src/

fetch:
	uv run python -m src.data.fetcher

train:
	uv run python -m src.models.trainer

predict:
	PYTHONPATH=. uv run python scripts/predict.py --symbol BTC-USDT

app:
	uv run streamlit run app.py

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache
