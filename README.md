# Market Data Pipeline + TensorFlow Forecasting Dashboard

**A modular, production-minded end-to-end market data pipeline with feature engineering and LSTM forecasting.**

Built as a generalized portfolio project to demonstrate clean engineering practices in financial time-series ML.

![Dashboard Preview](https://via.placeholder.com/800x400?text=Streamlit+Dashboard+Screenshot)  
*(Replace with actual screenshot/GIF after running the dashboard)*

## ✨ Features

- **Modular data fetching** using CCXT (supports 100+ exchanges via config)
- **Standard technical feature engineering** (RSI, Volume Ratio, Moving Averages, Volatility, etc.)
- **TensorFlow LSTM** time-series forecasting model with dynamic feature handling
- **Interactive Streamlit dashboard** for visualization and training
- **Reproducible pipeline** with proper configuration, logging, and error handling
- **Production-ready structure** (Docker-ready, CI/CD friendly, clean separation of concerns)

## 🛠 Tech Stack

- **Python 3.12**
- **CCXT** – Unified crypto exchange data fetching
- **TensorFlow / Keras** – Deep learning model
- **Pandas** – Data processing & feature engineering
- **Streamlit** – Interactive dashboard
- **Plotly** – Interactive visualizations
- **Pydantic** – Configuration management
- **UV** – Fast dependency management

## 🚀 Quick Start

### 1. Clone and install
```bash
git clone https://github.com/YOURUSERNAME/market-data-tf-pipeline.git
cd market-data-tf-pipeline
uv sync