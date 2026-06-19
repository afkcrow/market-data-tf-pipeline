"""
tests/unit/test_baseline.py
Tests for the naive persistence baseline used to sanity-check the LSTM.
"""

import numpy as np

from src.evaluation.baseline import (
    compare_to_persistence,
    directional_accuracy,
    model_mae,
    persistence_mae,
)


def test_persistence_mae_matches_absolute_diffs():
    actual = np.array([100.0, 102.0, 101.0, 105.0])
    # |102-100| + |101-102| + |105-101| = 2 + 1 + 4 = 7, over 3 steps
    assert persistence_mae(actual) == 7.0 / 3.0


def test_perfect_model_beats_persistence():
    actual = np.array([100.0, 102.0, 101.0, 105.0, 103.0])
    # A model that nails every next close has zero error.
    perfect = actual.copy()
    result = compare_to_persistence(perfect, actual)

    assert result["model_mae"] == 0.0
    assert result["persistence_mae"] > 0.0
    assert result["beats_persistence"] is True
    assert result["skill_score"] == 1.0


def test_persistence_model_scores_zero_skill():
    """A model that just echoes the previous close == persistence (skill ~ 0)."""
    actual = np.array([100.0, 102.0, 101.0, 105.0, 103.0])
    echo = np.concatenate([[actual[0]], actual[:-1]])  # predict prev close

    result = compare_to_persistence(echo, actual)
    assert result["model_mae"] == result["persistence_mae"]
    assert result["skill_score"] == 0.0
    assert result["beats_persistence"] is False


def test_directional_accuracy_perfect_and_inverse():
    actual = np.array([100.0, 101.0, 100.0, 102.0])
    # Forecast aligned so predicted direction always matches the real move.
    up_then_down = np.array([100.0, 105.0, 95.0, 110.0])
    assert directional_accuracy(up_then_down, actual) == 1.0

    # Always forecast a drop -> wrong on every up move.
    always_down = actual - 50.0
    acc = directional_accuracy(always_down, actual)
    assert 0.0 <= acc < 1.0


def test_model_mae_ignores_first_target():
    actual = np.array([10.0, 20.0, 30.0])
    predicted = np.array([0.0, 22.0, 27.0])  # first entry ignored by alignment
    # |22-20| + |27-30| = 2 + 3 = 5, over 2 steps
    assert model_mae(predicted, actual) == 2.5
