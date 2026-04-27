"""
Prophet-based forecasting for M5.

Public API:
    fit_prophet(series, dates, holidays_df, horizon=28) -> np.ndarray
    run_batch(sample_ids, sales_train, prices_df, calendar_df,
              changepoint_prior_scale=0.05, n_jobs=2, horizon=28)
        -> (preds array shape (n_sample, 28), metadata dict)

Design notes:
- Per-series fitting using Meta's Prophet, same 1k-sample convention as Day 3.
- Custom holiday DataFrame built from M5 calendar event_name_1 / event_name_2.
- Weekly seasonality enabled (Fourier order 3); yearly seasonality disabled
  (M5 training window is ~5 years but many series have < 2 years of non-zero
  history; yearly terms overfit on short series).
- changepoint_prior_scale is the key tuning knob: 0.01 (underfit), 0.05
  (default), 0.1 (more flexible trend).
- Sparse series (>80% zeros): zero forecast, Prophet skipped.
- Fallback on any fit exception: seasonal naive (7-day period).
- n_jobs=2 to avoid the SARIMA OOM pattern (each Prophet process is heavier
  than ETS due to Stan backend).
"""
from __future__ import annotations

import time
import warnings
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HORIZON = 28
ZERO_THRESHOLD = 0.80


# ── helpers ───────────────────────────────────────────────────────────────────

def _is_sparse(series: np.ndarray) -> bool:
    return (series == 0).mean() > ZERO_THRESHOLD


def _clip(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0.0, None)


def _seasonal_naive_fallback(series: np.ndarray, horizon: int, period: int = 7) -> np.ndarray:
    last = series[-period:]
    reps = (horizon + period - 1) // period
    return _clip(np.tile(last, reps)[:horizon])


# ── M5 holiday builder ────────────────────────────────────────────────────────

def build_m5_holidays(calendar_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a Prophet-compatible holidays DataFrame from the M5 calendar.

    Prophet expects columns: holiday (str), ds (datetime).
    Optional: lower_window, upper_window (int) — days around the event.

    M5 has event_name_1 / event_name_2.  We use both columns, drop nulls,
    and assign a window of [-1, 1] around each event (the day before and
    after are often affected by shopping behaviour).
    """
    rows = []
    for _, r in calendar_df.iterrows():
        ds = pd.to_datetime(r["date"])
        for col in ["event_name_1", "event_name_2"]:
            name = r.get(col)
            if pd.notna(name) and str(name).strip():
                rows.append({"holiday": str(name), "ds": ds,
                             "lower_window": -1, "upper_window": 1})
    if not rows:
        return pd.DataFrame(columns=["holiday", "ds"])
    return pd.DataFrame(rows).drop_duplicates()


# ── single-series fit ─────────────────────────────────────────────────────────

def fit_prophet(
    series: np.ndarray,
    dates: pd.DatetimeIndex,
    holidays_df: pd.DataFrame,
    horizon: int = HORIZON,
    changepoint_prior_scale: float = 0.05,
) -> np.ndarray:
    """
    Fit Prophet to one M5 series and return a (horizon,) forecast array.

    Parameters
    ----------
    series : training values aligned to `dates`
    dates  : DatetimeIndex of length len(series), daily frequency
    holidays_df : Prophet holidays DataFrame (from build_m5_holidays)
    changepoint_prior_scale : flexibility of trend changes
    """
    if _is_sparse(series):
        return np.zeros(horizon)

    try:
        from prophet import Prophet
    except ImportError:
        return _seasonal_naive_fallback(series, horizon)

    df = pd.DataFrame({"ds": dates, "y": series.astype(float)})

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = Prophet(
                changepoint_prior_scale=changepoint_prior_scale,
                weekly_seasonality=True,
                yearly_seasonality=False,
                daily_seasonality=False,
                seasonality_mode="multiplicative",
                holidays=holidays_df if len(holidays_df) > 0 else None,
            )
            m.fit(df)

            future = m.make_future_dataframe(periods=horizon, freq="D")
            forecast = m.predict(future)
            preds = forecast["yhat"].values[-horizon:]
            return _clip(preds.astype(np.float64))

    except Exception:
        return _seasonal_naive_fallback(series, horizon)


# ── single-series dispatcher for joblib ──────────────────────────────────────

def _fit_one(args: tuple) -> tuple:
    """Returns (idx, preds, is_fallback, is_zero, fit_time)."""
    (idx, series, dates, holidays_df, horizon, changepoint_prior_scale) = args

    t0 = time.time()

    if _is_sparse(series):
        return idx, np.zeros(horizon), False, True, time.time() - t0

    preds = fit_prophet(series, dates, holidays_df, horizon, changepoint_prior_scale)

    is_fallback = (preds == _seasonal_naive_fallback(series, horizon, 7)).all()
    is_zero = (preds == 0).all()
    return idx, preds, bool(is_fallback), bool(is_zero), time.time() - t0


# ── batch runner ──────────────────────────────────────────────────────────────

def run_batch(
    sample_ids: list,
    sales_train: pd.DataFrame,
    prices_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    changepoint_prior_scale: float = 0.05,
    n_jobs: int = 2,
    horizon: int = HORIZON,
    last_train_day: int = 1913,
) -> tuple[np.ndarray, dict]:
    """
    Run Prophet on all series in sample_ids.

    Returns
    -------
    preds : np.ndarray shape (len(sample_ids), horizon)
    meta  : dict with timing/fallback stats and hyperparameter record
    """
    from joblib import Parallel, delayed

    sales_sub = (
        sales_train[sales_train["id"].isin(sample_ids)]
        .set_index("id")
        .reindex(sample_ids)
        .reset_index()
    )

    train_cols = [f"d_{d}" for d in range(1, last_train_day + 1)
                  if f"d_{d}" in sales_sub.columns]

    # Build date index: M5 d_1 = 2011-01-29
    d1 = pd.Timestamp("2011-01-29")
    dates = pd.date_range(start=d1, periods=len(train_cols), freq="D")

    holidays_df = build_m5_holidays(calendar_df)

    args_list = []
    for i, row in enumerate(sales_sub.itertuples()):
        series = np.array([getattr(row, c) for c in train_cols], dtype=np.float64)
        args_list.append((i, series, dates, holidays_df, horizon, changepoint_prior_scale))

    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
        delayed(_fit_one)(a) for a in args_list
    )

    n = len(sample_ids)
    preds = np.zeros((n, horizon), dtype=np.float64)
    fit_times = []
    n_fallbacks = 0
    n_zero_forecasts = 0

    for idx, pred_arr, is_fallback, is_zero, fit_time in results:
        preds[idx] = pred_arr
        fit_times.append(fit_time)
        if is_fallback:
            n_fallbacks += 1
        if is_zero:
            n_zero_forecasts += 1

    meta = {
        "method": "prophet",
        "changepoint_prior_scale": changepoint_prior_scale,
        "n_series": n,
        "n_jobs": n_jobs,
        "n_fallbacks": n_fallbacks,
        "n_zero_forecasts": n_zero_forecasts,
        "fit_time_per_series_mean": float(np.mean(fit_times)),
        "fit_time_per_series_median": float(np.median(fit_times)),
        "fit_time_total": float(np.sum(fit_times)),
    }

    return preds, meta
