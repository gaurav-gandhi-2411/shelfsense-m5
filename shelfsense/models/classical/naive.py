"""
Naive and moving-average forecasting baselines for M5.

All methods accept a (n_series, n_train) float array and return (n_series, horizon).
"""
from __future__ import annotations

import numpy as np

HORIZON = 28


def naive_last(train: np.ndarray, horizon: int = HORIZON) -> np.ndarray:
    """Forecast = last observed value repeated for all horizon steps."""
    return np.tile(train[:, -1:], (1, horizon))


def seasonal_naive(
    train: np.ndarray,
    period: int,
    horizon: int = HORIZON,
) -> np.ndarray:
    """
    Cycle the last *period* observed values forward to fill the horizon.
    For period=7: repeats the last week's pattern.
    For period=28: repeats the last 28 days exactly.
    For period=365: uses the same-window from last year.
    """
    last = train[:, -period:]             # (n_series, period)
    reps = (horizon + period - 1) // period
    return np.tile(last, (1, reps))[:, :horizon]


def moving_average(
    train: np.ndarray,
    window: int,
    horizon: int = HORIZON,
) -> np.ndarray:
    """Forecast = mean of last *window* values, flat across all horizon steps."""
    mean_vals = train[:, -window:].mean(axis=1, keepdims=True).clip(min=0)
    return np.tile(mean_vals, (1, horizon))


# ── convenience registry ──────────────────────────────────────────────────────

def get_all_baselines(
    train: np.ndarray,
    horizon: int = HORIZON,
) -> dict[str, np.ndarray]:
    """Return all 7 baseline forecasts keyed by name."""
    return {
        "naive":              naive_last(train, horizon),
        "seasonal_naive_7":   seasonal_naive(train, period=7,   horizon=horizon),
        "seasonal_naive_28":  seasonal_naive(train, period=28,  horizon=horizon),
        "seasonal_naive_365": seasonal_naive(train, period=365, horizon=horizon),
        "moving_avg_7":       moving_average(train, window=7,   horizon=horizon),
        "moving_avg_28":      moving_average(train, window=28,  horizon=horizon),
        "moving_avg_90":      moving_average(train, window=90,  horizon=horizon),
    }
