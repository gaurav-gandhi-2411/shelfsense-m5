"""
Recursive multi-step forecasting for the M5 evaluation period (d_1942–d_1969).

The key function predict_recursive() generates predictions one day at a time,
feeding each prediction back into lag/rolling feature computation for the next day.

Usage
-----
from models.recursive_forecast import (
    build_sales_buffer, build_eval_price_features, predict_recursive
)
"""
from __future__ import annotations

import time
import numpy as np
import pandas as pd

# Must match 06_train_lightgbm.py / features/lags.py / features/rolling.py
LAGS    = [7, 14, 28, 56]
WINDOWS = [7, 28, 56, 180]
BUFFER_SIZE = 200   # covers max(WINDOWS)=180; 200 provides a small safety margin

CAL_FEATURE_COLS = [
    "weekday", "month", "quarter", "year", "day_of_month", "week_of_year",
    "is_weekend", "is_holiday", "is_snap_ca", "is_snap_tx", "is_snap_wi",
    "days_since_event", "days_until_next_event",
]
PRICE_FEATURE_COLS = [
    "sell_price", "price_change_pct", "price_relative_mean",
    "price_volatility", "has_price_change",
]
LAG_FEATURES = [f"lag_{l}" for l in LAGS]
# Must match training column order: for each window W, stats are mean/std/min/max
ROLL_FEATURES = [
    f"roll_{stat}_{w}"
    for w in WINDOWS
    for stat in ["mean", "std", "min", "max"]
]
NUM_FEATURES = CAL_FEATURE_COLS + PRICE_FEATURE_COLS + LAG_FEATURES + ROLL_FEATURES
CAT_COLS = ["cat_id", "dept_id", "store_id", "state_id"]
ALL_FEATURES = NUM_FEATURES + CAT_COLS  # 42 total (38 num + 4 cat)


def build_sales_buffer(
    sales_eval: pd.DataFrame,
    series_order: np.ndarray,
    last_day: int = 1941,
    buffer_size: int = BUFFER_SIZE,
) -> np.ndarray:
    """
    Extract the last `buffer_size` days of actual sales.

    Returns
    -------
    (n_series, buffer_size) float32 array.
    buffer[:, -1] = sales[last_day]  (most recent)
    buffer[:, 0]  = sales[last_day - buffer_size + 1]  (oldest)

    Pre-launch NaN (series not yet in market) → 0.
    """
    first_day = last_day - buffer_size + 1
    day_cols = [f"d_{d}" for d in range(first_day, last_day + 1)]
    arr = (
        sales_eval
        .set_index("id")
        .reindex(series_order)[day_cols]
        .values
        .astype(np.float32)
    )
    return np.nan_to_num(arr, nan=0.0)


def build_eval_price_features(
    series_meta: pd.DataFrame,
    price_lookup: pd.DataFrame,
    cal_lookup: pd.DataFrame,
    eval_start: int = 1942,
    eval_end: int = 1969,
) -> dict:
    """
    Pre-compute per-series price features for every day in [eval_start, eval_end].

    Returns
    -------
    dict mapping d_num → (n_series, 5) float32 array in PRICE_FEATURE_COLS order.
    Missing prices (items with no price data for that week) → 0.
    """
    d_to_wk = {
        d: int(cal_lookup.loc[f"d_{d}", "wm_yr_wk"])
        for d in range(eval_start, eval_end + 1)
    }
    unique_wks = sorted(set(d_to_wk.values()))

    wk_arrays: dict[int, np.ndarray] = {}
    for wk in unique_wks:
        wk_prices = (
            price_lookup[price_lookup["wm_yr_wk"] == wk]
            .set_index(["item_id", "store_id"])[PRICE_FEATURE_COLS]
        )
        merged = series_meta[["item_id", "store_id"]].merge(
            wk_prices.reset_index(), on=["item_id", "store_id"], how="left"
        )
        arr = np.nan_to_num(merged[PRICE_FEATURE_COLS].values.astype(np.float32), nan=0.0)
        wk_arrays[wk] = arr

    return {d: wk_arrays[d_to_wk[d]] for d in range(eval_start, eval_end + 1)}


def predict_recursive(
    model,
    sales_buffer: np.ndarray,
    series_meta: pd.DataFrame,
    cal_lookup: pd.DataFrame,
    price_by_d: dict,
    cat_dtypes: dict,
    start_day: int,
    end_day: int,
    all_features: list = ALL_FEATURES,
    verbose: bool = True,
) -> np.ndarray:
    """
    Recursive multi-step forecast over [start_day, end_day].

    At each step:
      1. Compute lag/rolling features from the sales buffer.
      2. Look up calendar and price features for the current day.
      3. Predict with model.predict() for all series simultaneously.
      4. Clip predictions to ≥ 0 and append to buffer.

    Buffer convention
    -----------------
    buffer[:, -1] = sales[start_day - 1] on entry.
    After predicting day d:  buffer[:, -1] = pred[d].

    Lag check (for day d):
      lag_7(d) = sales[d-7] = buffer[:, -7]  ✓  (7 positions from the right)
    Rolling check (for day d):
      roll_mean_7(d) = mean(sales[d-7..d-1]) = buffer[:, -7:].mean()  ✓
      (matches features/rolling.py which uses shift(1) before rolling)

    Parameters
    ----------
    sales_buffer : (n_series, buffer_size) float32 — must have buffer_size ≥ 180.
    series_meta  : DataFrame with columns id, item_id, dept_id, cat_id, store_id, state_id.
                   Row i corresponds to buffer[i].
    cat_dtypes   : {col: CategoricalDtype} — must match training (use hierarchy.CAT_DTYPES).
    all_features : Feature column list in training order (default = ALL_FEATURES here).

    Returns
    -------
    (n_series, n_steps) float32 where n_steps = end_day - start_day + 1.
    """
    n_series = len(series_meta)
    n_steps  = end_day - start_day + 1
    preds    = np.zeros((n_series, n_steps), dtype=np.float32)
    buf      = sales_buffer.copy()

    t_total = time.time()
    for step, day in enumerate(range(start_day, end_day + 1)):
        t0 = time.time()

        # ── calendar (broadcast across all series) ──────────────────────────────
        cal_row = cal_lookup.loc[f"d_{day}", CAL_FEATURE_COLS].values.astype(np.float32)
        cal_mat = np.tile(cal_row, (n_series, 1))           # (n_series, 13)

        # ── price (per series, pre-computed) ────────────────────────────────────
        price_mat = price_by_d[day]                          # (n_series, 5)

        # ── lag features: buf[:, -k] = sales[day - k] ──────────────────────────
        lag_mat = np.column_stack([buf[:, -lag] for lag in LAGS])  # (n_series, 4)

        # ── rolling: shift-1 window = buf[:, -w:] = [day-w .. day-1] ───────────
        roll_cols = []
        for w in WINDOWS:
            sl = buf[:, -w:]                                 # (n_series, w)
            roll_cols.append(sl.mean(axis=1))
            std_ = np.where(sl.shape[1] > 1, sl.std(axis=1, ddof=1), 0.0)
            roll_cols.append(np.nan_to_num(std_, nan=0.0))
            roll_cols.append(sl.min(axis=1))
            roll_cols.append(sl.max(axis=1))
        roll_mat = np.column_stack(roll_cols)                # (n_series, 16)

        # ── assemble numerical block (38 columns, training order) ───────────────
        num_mat  = np.hstack([cal_mat, price_mat, lag_mat, roll_mat]).astype(np.float32)
        feat_df  = pd.DataFrame(num_mat, columns=NUM_FEATURES)

        # ── categorical features ─────────────────────────────────────────────────
        for col in CAT_COLS:
            str_vals = series_meta[col].astype(str).values
            feat_df[col] = pd.Categorical(
                str_vals, categories=cat_dtypes[col].categories, ordered=False
            )

        # ── reorder to match training ALL_FEATURES ───────────────────────────────
        feat_df = feat_df[all_features]

        # ── predict ───────────────────────────────────────────────────────────────
        step_preds = np.clip(model.predict(feat_df), 0.0, None).astype(np.float32)
        preds[:, step] = step_preds

        # ── update buffer: shift left 1, append prediction ───────────────────────
        buf = np.roll(buf, -1, axis=1)
        buf[:, -1] = step_preds

        if verbose:
            print(f"    d_{day}  ({time.time()-t0:.1f}s)", end="  ", flush=True)
            if (step + 1) % 7 == 0 or step == n_steps - 1:
                print(flush=True)

    if verbose:
        print(f"    Total: {time.time()-t_total:.1f}s", flush=True)
    return preds
