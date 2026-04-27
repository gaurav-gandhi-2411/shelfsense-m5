"""
Classical statistical forecasting methods for M5.

Public API:
    fit_ets(series, horizon=28) -> np.ndarray
    fit_arima(series, horizon=28) -> np.ndarray
    fit_sarima(series, horizon=28) -> np.ndarray
    fit_sarimax(series, exog_train, exog_future, horizon=28) -> np.ndarray
    run_batch(method, sample_ids, sales_train, prices_df, calendar_df, n_jobs=4, horizon=28)
        -> (preds array shape (n_sample, 28), metadata dict)
"""
from __future__ import annotations

import sys
import os
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HORIZON = 28
ZERO_THRESHOLD = 0.80   # series with >80% zeros => zero forecast


# ── helpers ───────────────────────────────────────────────────────────────────

def _is_sparse(series: pd.Series) -> bool:
    """Return True if >80% of values are zero."""
    return (series == 0).mean() > ZERO_THRESHOLD


def _clip(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0.0, None)


def _seasonal_naive_fallback(series: pd.Series, horizon: int, period: int = 7) -> np.ndarray:
    vals = series.values.astype(float)
    last = vals[-period:]
    reps = (horizon + period - 1) // period
    return _clip(np.tile(last, reps)[:horizon])


def _last_value_fallback(series: pd.Series, horizon: int) -> np.ndarray:
    return _clip(np.full(horizon, series.iloc[-1], dtype=float))


# ── ETS ───────────────────────────────────────────────────────────────────────

def fit_ets(series: pd.Series, horizon: int = HORIZON) -> np.ndarray:
    """
    Exponential smoothing via statsmodels HoltWinters.
    Primary: trend='add', seasonal='add', seasonal_periods=7
    Fallback: simple exponential smoothing (no trend/seasonal)
    Clip to >= 0.
    """
    if _is_sparse(series):
        return np.zeros(horizon)

    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    vals = series.values.astype(float)

    # Primary: additive trend + additive weekly seasonality
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ExponentialSmoothing(
                vals,
                trend="add",
                seasonal="add",
                seasonal_periods=7,
                initialization_method="estimated",
            )
            fit = model.fit(optimized=True)
            return _clip(fit.forecast(horizon))
    except Exception:
        pass

    # Fallback: simple exponential smoothing
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ExponentialSmoothing(
                vals,
                trend=None,
                seasonal=None,
                initialization_method="estimated",
            )
            fit = model.fit(optimized=True)
            return _clip(fit.forecast(horizon))
    except Exception:
        pass

    return _last_value_fallback(series, horizon)


# ── ARIMA ─────────────────────────────────────────────────────────────────────

def fit_arima(series: pd.Series, horizon: int = HORIZON) -> np.ndarray:
    """
    Auto ARIMA (non-seasonal) via pmdarima.
    Falls back to naive last-value on failure.
    """
    if _is_sparse(series):
        return np.zeros(horizon)

    import pmdarima as pm

    vals = series.values.astype(float)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = pm.auto_arima(
                vals,
                max_p=3, max_q=3, max_d=1,
                seasonal=False,
                error_action="ignore",
                suppress_warnings=True,
                stepwise=True,
            )
            preds = model.predict(n_periods=horizon)
            return _clip(np.array(preds, dtype=float))
    except Exception:
        pass

    return _last_value_fallback(series, horizon)


# ── SARIMA ────────────────────────────────────────────────────────────────────

def fit_sarima(series: pd.Series, horizon: int = HORIZON) -> np.ndarray:
    """
    Seasonal ARIMA via pmdarima with m=7.
    Falls back to seasonal naive (7-day) on failure.
    """
    if _is_sparse(series):
        return np.zeros(horizon)

    import pmdarima as pm

    vals = series.values.astype(float)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = pm.auto_arima(
                vals,
                max_p=2, max_q=2, max_d=1,
                seasonal=True, m=7,
                D=1, max_P=1, max_Q=1,
                error_action="ignore",
                suppress_warnings=True,
                stepwise=True,
            )
            preds = model.predict(n_periods=horizon)
            return _clip(np.array(preds, dtype=float))
    except Exception:
        pass

    return _seasonal_naive_fallback(series, horizon, period=7)


# ── SARIMAX ───────────────────────────────────────────────────────────────────

def fit_sarimax(
    series: pd.Series,
    exog_train: pd.DataFrame,
    exog_future: pd.DataFrame,
    horizon: int = HORIZON,
) -> np.ndarray:
    """
    SARIMA with exogenous features via pmdarima.
    Falls back to fit_sarima on failure.
    """
    if _is_sparse(series):
        return np.zeros(horizon)

    import pmdarima as pm

    vals = series.values.astype(float)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = pm.auto_arima(
                vals,
                exogenous=exog_train.values,
                max_p=2, max_q=2, max_d=1,
                seasonal=True, m=7,
                D=1, max_P=1, max_Q=1,
                error_action="ignore",
                suppress_warnings=True,
                stepwise=True,
            )
            preds = model.predict(n_periods=horizon, exogenous=exog_future.values)
            return _clip(np.array(preds, dtype=float))
    except Exception:
        pass

    return fit_sarima(series, horizon)


# ── exogenous feature builder ─────────────────────────────────────────────────

def build_exog(
    series_id: str,
    state_id: str,
    prices_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    item_id: str,
    store_id: str,
    last_train_day: int = 1913,
    horizon: int = HORIZON,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build exogenous feature DataFrames for training and forecast horizon.

    Features:
      is_holiday : 1 if event_name_1 is not null
      snap       : state-specific SNAP flag
      is_weekend : 1 if wday in {1,7}
      month      : 1-12
      sell_price : last known sell price (forward-filled)
    """
    # Training days: d_1 .. d_{last_train_day}
    train_day_cols = [f"d_{d}" for d in range(1, last_train_day + 1)]
    future_day_cols = [f"d_{d}" for d in range(last_train_day + 1, last_train_day + horizon + 1)]

    cal = calendar_df.copy()
    snap_col = f"snap_{state_id}"

    def _make_features(day_cols):
        rows = cal[cal["d"].isin(day_cols)].set_index("d").reindex(day_cols)
        feat = pd.DataFrame(index=day_cols)
        feat["is_holiday"] = rows["event_name_1"].notna().astype(int).values
        feat["snap"] = rows[snap_col].fillna(0).astype(int).values if snap_col in rows.columns else 0
        feat["is_weekend"] = rows["wday"].isin([1, 7]).astype(int).values
        feat["month"] = rows["month"].fillna(rows["month"].mode()[0] if len(rows["month"].dropna()) > 0 else 1).astype(int).values
        return feat

    exog_train = _make_features(train_day_cols)
    exog_future = _make_features(future_day_cols)

    # sell_price: merge by wm_yr_wk
    cal_indexed = cal.set_index("d")
    price_sub = prices_df[
        (prices_df["store_id"] == store_id) & (prices_df["item_id"] == item_id)
    ][["wm_yr_wk", "sell_price"]].copy()

    def _add_price(feat, day_cols):
        wks = cal_indexed.reindex(day_cols)["wm_yr_wk"].values
        feat = feat.copy()
        feat["wm_yr_wk"] = wks
        feat = feat.merge(price_sub, on="wm_yr_wk", how="left")
        feat["sell_price"] = feat["sell_price"].ffill().bfill().fillna(0.0)
        feat = feat.drop(columns=["wm_yr_wk"])
        return feat

    exog_train = _add_price(exog_train, train_day_cols)
    exog_future = _add_price(exog_future, future_day_cols)

    return exog_train, exog_future


# ── single-series dispatcher ──────────────────────────────────────────────────

def _fit_one(
    args: tuple,
) -> tuple[int, np.ndarray, bool, bool, float]:
    """
    Fit one series. Returns (idx, preds, is_fallback, is_zero, fit_time).
    Catches all exceptions.
    """
    (
        idx, method, series, row_meta,
        prices_df, calendar_df, horizon, last_train_day
    ) = args

    t0 = time.time()
    is_fallback = False
    is_zero = False

    try:
        if method == "ets":
            preds = fit_ets(series, horizon)
        elif method == "arima":
            preds = fit_arima(series, horizon)
        elif method == "sarima":
            preds = fit_sarima(series, horizon)
        elif method == "sarimax":
            exog_train, exog_future = build_exog(
                series_id=row_meta["id"],
                state_id=row_meta["state_id"],
                prices_df=prices_df,
                calendar_df=calendar_df,
                item_id=row_meta["item_id"],
                store_id=row_meta["store_id"],
                last_train_day=last_train_day,
                horizon=horizon,
            )
            preds = fit_sarimax(series, exog_train, exog_future, horizon)
        else:
            raise ValueError(f"Unknown method: {method}")

        if (preds == 0).all():
            is_zero = True

    except Exception as e:
        is_fallback = True
        preds = np.zeros(horizon)

    fit_time = time.time() - t0
    return idx, preds, is_fallback, is_zero, fit_time


# ── batch runner ──────────────────────────────────────────────────────────────

def run_batch(
    method: str,
    sample_ids: list,
    sales_train: pd.DataFrame,
    prices_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    n_jobs: int = 4,
    horizon: int = HORIZON,
    last_train_day: int = 1913,
) -> tuple[np.ndarray, dict]:
    """
    Run one classical method on all series in sample_ids.

    Returns
    -------
    preds : np.ndarray shape (len(sample_ids), horizon)
    meta  : dict with timing/fallback stats
    """
    from joblib import Parallel, delayed

    # Filter sales_train to sample
    sales_sub = sales_train[sales_train["id"].isin(sample_ids)].copy()
    # Reindex to match sample_ids order
    sales_sub = sales_sub.set_index("id").reindex(sample_ids).reset_index()

    train_cols = [c for c in (f"d_{d}" for d in range(1, last_train_day + 1))
                  if c in sales_sub.columns]

    meta_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]

    args_list = []
    for i, row in enumerate(sales_sub.itertuples()):
        row_meta = {c: getattr(row, c) for c in meta_cols}
        series = pd.Series(
            [getattr(row, c) for c in train_cols],
            name=row_meta["id"],
        )
        args_list.append((
            i, method, series, row_meta,
            prices_df, calendar_df, horizon, last_train_day
        ))

    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
        delayed(_fit_one)(a) for a in args_list
    )

    n = len(sample_ids)
    preds = np.zeros((n, horizon), dtype=np.float32)
    fit_times = []
    n_fallbacks = 0
    n_zero_forecasts = 0

    for idx, pred_arr, is_fallback, is_zero, fit_time in results:
        preds[idx] = pred_arr.astype(np.float32)
        fit_times.append(fit_time)
        if is_fallback:
            n_fallbacks += 1
        if is_zero:
            n_zero_forecasts += 1

    metadata = {
        "method": method,
        "n_series": n,
        "n_fallbacks": n_fallbacks,
        "n_zero_forecasts": n_zero_forecasts,
        "fit_time_per_series_mean": float(np.mean(fit_times)),
        "fit_time_per_series_median": float(np.median(fit_times)),
        "fit_time_total": float(np.sum(fit_times)),
    }

    return preds, metadata
