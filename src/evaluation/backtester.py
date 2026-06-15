"""
src/evaluation/backtester.py
Directional backtester: go long when model predicts price increase, flat otherwise.
"""

from __future__ import annotations

import re

import numpy as np

_MINUTES_PER_UNIT = {"m": 1, "h": 60, "d": 60 * 24, "w": 60 * 24 * 7}
_MINUTES_PER_YEAR = 365 * 24 * 60


def periods_per_year(timeframe: str) -> float:
    """
    Convert a CCXT-style timeframe (e.g. '15m', '4h', '1d') to the number of
    such periods in a calendar year. Used for Sharpe annualization on crypto
    (24/7 markets — no business-day adjustment).
    """
    match = re.fullmatch(r"(\d+)([mhdw])", timeframe.strip().lower())
    if not match:
        raise ValueError(f"Unrecognized timeframe: {timeframe!r}")
    n, unit = int(match.group(1)), match.group(2)
    return _MINUTES_PER_YEAR / (n * _MINUTES_PER_UNIT[unit])


class Backtester:
    def __init__(
        self,
        initial_capital: float = 10_000.0,
        fee_bps: float = 10.0,
        timeframe: str = "1d",
    ):
        self.initial_capital = initial_capital
        self.fee = fee_bps / 10_000
        self.periods_per_year = periods_per_year(timeframe)

    def run(self, actual_prices: np.ndarray, predicted_prices: np.ndarray) -> dict:
        """
        Simulate a long-only strategy driven by model direction.
        predicted_prices[i] is the forecast for the close at step i+1.
        Returns performance metrics and the full equity curve.
        """
        if len(actual_prices) != len(predicted_prices):
            raise ValueError("actual_prices and predicted_prices must be the same length")

        capital = self.initial_capital
        position = 0.0
        equity_curve = [capital]

        for i in range(len(actual_prices) - 1):
            current_price = actual_prices[i]
            next_price = actual_prices[i + 1]
            long_signal = predicted_prices[i] > current_price

            if long_signal and position == 0.0:
                position = (capital / current_price) * (1 - self.fee)
                capital = 0.0
            elif not long_signal and position > 0.0:
                capital = position * current_price * (1 - self.fee)
                position = 0.0

            equity_curve.append(capital + position * next_price)

        # Close any remaining position at the final price.
        if position > 0.0:
            capital = position * actual_prices[-1] * (1 - self.fee)
            equity_curve[-1] = capital

        equity = np.array(equity_curve)
        returns = np.diff(equity) / equity[:-1]

        total_return = (equity[-1] - self.initial_capital) / self.initial_capital
        if returns.std() > 0:
            sharpe = float(returns.mean() / returns.std() * np.sqrt(self.periods_per_year))
        else:
            sharpe = 0.0
        max_dd = self._max_drawdown(equity)

        return {
            "total_return": total_return,
            "final_capital": float(equity[-1]),
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "equity_curve": equity,
        }

    @staticmethod
    def _max_drawdown(equity: np.ndarray) -> float:
        peak = equity[0]
        max_dd = 0.0
        for val in equity:
            if val > peak:
                peak = val
            dd = (peak - val) / peak
            if dd > max_dd:
                max_dd = dd
        return float(max_dd)
