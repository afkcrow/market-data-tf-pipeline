"""
src/evaluation/baseline.py
Naive persistence baseline: forecast that the next close equals the current close.

On noisy financial series this is the honest yardstick — an LSTM that can't beat
persistence has mostly learned "the next bar looks like this one". These helpers
report the model's error and directional hit-rate against that baseline on
identical next-close targets, so the comparison is apples-to-apples.
"""

from __future__ import annotations

import numpy as np


def persistence_mae(actual: np.ndarray) -> float:
    """MAE of forecasting next close = current close, over the given series."""
    actual = np.asarray(actual, dtype=float)
    if len(actual) < 2:
        return 0.0
    return float(np.mean(np.abs(actual[1:] - actual[:-1])))


def model_mae(predicted: np.ndarray, actual: np.ndarray) -> float:
    """MAE of the model's next-close forecast on the same aligned targets."""
    predicted = np.asarray(predicted, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if len(actual) < 2:
        return 0.0
    return float(np.mean(np.abs(predicted[1:] - actual[1:])))


def directional_accuracy(predicted: np.ndarray, actual: np.ndarray) -> float:
    """
    Fraction of steps where the model gets the direction of the next move right.
    Predicted direction is sign(forecast_next - current_close); actual direction
    is sign(actual_next - current_close). Persistence has no edge here — it always
    predicts no change — so this metric is reported for the model only.
    """
    predicted = np.asarray(predicted, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if len(actual) < 2:
        return 0.0
    pred_dir = np.sign(predicted[1:] - actual[:-1])
    actual_dir = np.sign(actual[1:] - actual[:-1])
    return float(np.mean(pred_dir == actual_dir))


def compare_to_persistence(predicted: np.ndarray, actual: np.ndarray) -> dict:
    """
    Model vs. naive persistence on identical next-close targets.

    `skill_score` is 1 - model_mae / persistence_mae: positive means the model
    beats persistence, zero means it matches it, negative means it's worse.
    """
    p_mae = persistence_mae(actual)
    m_mae = model_mae(predicted, actual)
    return {
        "model_mae": m_mae,
        "persistence_mae": p_mae,
        "beats_persistence": m_mae < p_mae,
        "skill_score": float(1 - m_mae / p_mae) if p_mae > 0 else 0.0,
        "directional_accuracy": directional_accuracy(predicted, actual),
    }
