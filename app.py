"""
app.py
Professional Interactive Dashboard for Market Data Pipeline
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import asyncio

from src.data.fetcher import MarketDataFetcher, FetchConfig
from src.features.technical import add_all_features
from src.models.trainer import PipelineTrainer

# Page Config
st.set_page_config(page_title="Market ML Pipeline", layout="wide", page_icon="📈")
st.title("📈 Market Data Pipeline + TensorFlow Forecasting")
st.markdown("**Generalized Portfolio Project** — Demonstrating Clean MLOps & Time-Series ML")

# Sidebar
with st.sidebar:
    st.header("Controls")
    timeframe = st.selectbox("Timeframe", ["15m", "4h", "1d"], index=0)
    symbol = st.selectbox("Symbol", ["BTC/USDT", "ETH/USDT"], index=0)
    
    col1, col2 = st.columns(2)
    with col1:
        fetch_btn = st.button("🔄 Fetch Data", use_container_width=True)
    with col2:
        train_btn = st.button("🚀 Train Model", use_container_width=True)

# Main Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Market Data", "📉 Technical Features", "🔮 LSTM Forecast", "ℹ️ About Project"])

with tab1:
    st.subheader(f"{symbol} Market Data ({timeframe})")
    if fetch_btn:
        with st.spinner("Fetching latest data..."):
            try:
                config = FetchConfig(timeframe=timeframe, symbols=[symbol])
                fetcher = MarketDataFetcher(config)
                asyncio.run(fetcher.update_all_symbols())
                st.success("✅ Data updated successfully!")
            except Exception as e:
                st.error(f"Fetch failed: {e}")

    try:
        trainer = PipelineTrainer(data_dir=Path("data/raw"), timeframe=timeframe)
        df = trainer.load_and_prepare_data(symbol.replace("/", "-"))
        
        st.dataframe(df.tail(10), use_container_width=True)
        
        fig = px.line(df, x=df.index, y='close', title=f"{symbol} Price History")
        st.plotly_chart(fig, use_container_width=True)
        
    except FileNotFoundError:
        st.info("No data yet. Click 'Fetch Data' to download candles.")

with tab2:
    st.subheader("Technical Indicators")
    try:
        df = trainer.load_and_prepare_data(symbol.replace("/", "-"))
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(px.line(df, x=df.index, y='rsi', title="RSI"), use_container_width=True)
            st.plotly_chart(px.line(df, x=df.index, y='volume_ratio_ma', title="Volume Ratio"), use_container_width=True)
        with col2:
            st.plotly_chart(px.line(df, x=df.index, y='volatility_20', title="Volatility"), use_container_width=True)
            st.plotly_chart(px.line(df, x=df.index, y='sma_20', title="SMA 20"), use_container_width=True)
    except:
        st.warning("Fetch data first to see indicators.")

with tab3:
    st.subheader("LSTM Model Training & Prediction")
    if train_btn:
        with st.spinner("Training LSTM model... (this may take a moment)"):
            try:
                trainer = PipelineTrainer(data_dir=Path("data/raw"), timeframe=timeframe)
                results = trainer.train_model(symbol=symbol.replace("/", "-"), epochs=12)
                st.success(f"✅ Model trained! Test MAE: **{results['test_mae']:.4f}**")
                st.balloons()
            except Exception as e:
                st.error(f"Training error: {e}")

with tab4:
    st.subheader("About This Project")
    st.markdown("""
    This is a **generalized portfolio project** built to demonstrate:
    - Clean, modular MLOps architecture
    - End-to-end time-series pipeline (fetch → features → model → visualization)
    - Production-ready practices (config, logging, testing, Docker-ready)
    - Domain knowledge in crypto market data
    
    **No proprietary alpha** is included.
    """)

st.caption("Self-taught ML + Blockchain Portfolio Project | 2026")