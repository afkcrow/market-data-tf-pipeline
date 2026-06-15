"""
app.py
Interactive dashboard for the market data + LSTM forecasting pipeline.
"""

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config.settings import FetchConfig
from src.data.fetcher import MarketDataFetcher
from src.evaluation.backtester import Backtester
from src.models.trainer import PipelineTrainer

logger = logging.getLogger(__name__)

# When APP_PASSWORD is set, expensive actions (fetch/train) require it. Unset = open (local dev).
APP_PASSWORD = os.environ.get("APP_PASSWORD")


def _authorized() -> bool:
    if not APP_PASSWORD:
        return True
    return st.session_state.get("_auth_pw", "") == APP_PASSWORD


st.set_page_config(page_title="Market ML Pipeline", layout="wide", page_icon="📈")
st.title("Market Data Pipeline + TensorFlow Forecasting")
st.markdown("**Portfolio Project** — End-to-end time-series ML with LSTM forecasting")

with st.sidebar:
    st.header("Controls")
    timeframe = st.selectbox("Timeframe", ["15m", "4h", "1d"], index=0)
    symbol = st.selectbox("Symbol", ["BTC/USDT", "ETH/USDT"], index=0)

    col1, col2 = st.columns(2)
    with col1:
        fetch_btn = st.button("Fetch Data", use_container_width=True)
    with col2:
        train_btn = st.button("Train Model", use_container_width=True)

    if APP_PASSWORD:
        st.text_input("Access password", type="password", key="_auth_pw")

tab1, tab2, tab3, tab4 = st.tabs(["Market Data", "Technical Features", "LSTM Forecast", "About"])

trainer = PipelineTrainer(data_dir=Path("data/raw"), timeframe=timeframe)
safe_symbol = symbol.replace("/", "-")


def _try_load() -> pd.DataFrame | None:
    try:
        return trainer.load_and_prepare_data(safe_symbol)
    except FileNotFoundError:
        return None


with tab1:
    st.subheader(f"{symbol} Market Data ({timeframe})")
    if fetch_btn and not _authorized():
        st.error("Unauthorized — enter the access password in the sidebar.")
    elif fetch_btn:
        with st.spinner("Fetching latest data..."):
            try:
                config = FetchConfig(timeframe=timeframe, symbols=[symbol])
                fetcher = MarketDataFetcher(config)
                # Run in a thread so asyncio.run() always gets a fresh event loop,
                # avoiding conflicts with Streamlit's own internal loop.
                with ThreadPoolExecutor(max_workers=1) as pool:
                    pool.submit(asyncio.run, fetcher.update_all_symbols()).result()
                st.success("Data updated successfully.")
            except Exception:
                logger.exception("Data fetch failed")
                st.error("Fetch failed. Check the server logs for details.")

    df = _try_load()
    if df is None:
        st.info("No data yet. Click 'Fetch Data' to download candles.")
    else:
        st.dataframe(df.tail(10), use_container_width=True)
        fig = px.line(df, x=df.index, y="close", title=f"{symbol} Price History")
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Technical Indicators")
    df = _try_load()
    if df is None:
        st.warning("Fetch data first to see indicators.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(px.line(df, x=df.index, y="rsi", title="RSI"), use_container_width=True)
            st.plotly_chart(
                px.line(df, x=df.index, y="volume_ratio_ma", title="Volume Ratio MA"),
                use_container_width=True,
            )
        with col2:
            st.plotly_chart(
                px.line(df, x=df.index, y="volatility_20", title="Rolling Volatility (20)"),
                use_container_width=True,
            )
            st.plotly_chart(
                px.line(df, x=df.index, y="sma_20", title="SMA 20"),
                use_container_width=True,
            )

with tab3:
    st.subheader("LSTM Model Training & Prediction")
    if train_btn and not _authorized():
        st.error("Unauthorized — enter the access password in the sidebar.")
    elif train_btn:
        with st.spinner("Training LSTM model... (this may take a moment)"):
            try:
                results = trainer.train_model(symbol=safe_symbol, epochs=12)

                test_actual = np.array(results["test_actual"])
                test_preds = np.array(results["test_predictions"])

                bt = Backtester(initial_capital=10_000.0, fee_bps=10.0, timeframe=timeframe)
                bt_results = bt.run(test_actual, test_preds)

                st.session_state["forecast"] = {
                    "mae": results["test_mae"],
                    "predictions": test_preds.tolist(),
                    "actual": test_actual.tolist(),
                    "bt": bt_results,
                    "symbol": safe_symbol,
                }
                st.success(f"Model trained. Test MAE: **${results['test_mae']:.2f}**")
                st.balloons()
            except FileNotFoundError:
                st.error("No data found for this symbol. Click 'Fetch Data' first.")
            except Exception:
                logger.exception("Model training failed")
                st.error("Training failed. Check the server logs for details.")

    if "forecast" in st.session_state:
        fr = st.session_state["forecast"]
        bt = fr["bt"]

        st.markdown("#### Backtest Results (directional strategy, 10 bps fee)")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Test MAE", f"${fr['mae']:,.2f}")
        m2.metric("Total Return", f"{bt['total_return']:.1%}")
        m3.metric("Sharpe Ratio", f"{bt['sharpe_ratio']:.2f}")
        m4.metric("Max Drawdown", f"{bt['max_drawdown']:.1%}")

        st.markdown("#### Predicted vs Actual (test set)")
        n = len(fr["actual"])
        pred_df = pd.DataFrame(
            {"Actual": fr["actual"], "Predicted": fr["predictions"]},
            index=range(n),
        )
        fig_pred = go.Figure()
        fig_pred.add_trace(
            go.Scatter(y=pred_df["Actual"], name="Actual", line=dict(color="#636EFA"))
        )
        fig_pred.add_trace(
            go.Scatter(
                y=pred_df["Predicted"], name="Predicted", line=dict(color="#EF553B", dash="dash")
            )
        )
        fig_pred.update_layout(
            title=f"{fr['symbol']} — Test Set: Predicted vs Actual Close",
            xaxis_title="Time step",
            yaxis_title="Price (USD)",
        )
        st.plotly_chart(fig_pred, use_container_width=True)

        st.markdown("#### Strategy Equity Curve")
        equity = bt["equity_curve"]
        eq_df = pd.DataFrame({"Equity ($)": equity}, index=range(len(equity)))
        fig_eq = px.line(eq_df, y="Equity ($)", title="Directional Strategy Equity Curve")
        fig_eq.add_hline(
            y=10_000, line_dash="dot", line_color="gray", annotation_text="Initial capital"
        )
        st.plotly_chart(fig_eq, use_container_width=True)
    else:
        st.info("Train a model using the sidebar button to see predictions and backtest results.")

with tab4:
    st.subheader("About This Project")
    st.markdown(
        """
        This is a **portfolio project** built to demonstrate:
        - Clean, modular MLOps architecture
        - End-to-end time-series pipeline (fetch → features → model → visualization)
        - Production-ready practices (config, logging, testing, Docker-ready)
        - Domain knowledge in crypto market data

        **No proprietary alpha** is included — this is a transparent reference implementation.

        **Stack:** Python 3.12 · CCXT · TensorFlow/Keras · pandas · scikit-learn · Pydantic · Streamlit · Plotly · uv · ruff · pytest
        """
    )

st.caption("ML Engineering Portfolio Project | 2026")
