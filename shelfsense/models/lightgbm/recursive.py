"""
Recursive multi-step forecaster.

Single entry point: predict_horizon(model, history_df, calendar_df, prices_df).

Key improvements over v1:
  - Exact day-index lookup for lags and rolling windows via searchsorted.
    No buffer-offset arithmetic; off-by-one errors are structurally impossible.
  - Rolling window selects only days in [d-w, d-1] by boolean mask over day_cols array.
  - Single unified API — no separate build_sales_buffer / build_eval_price_features calls.
  - Rolling std falls back to 0 when window has < 2 values (correct; matches NaN fill in training).

Expected recursive-vs-single-step gap:
  ~8-12% over 28 steps on M5 sparse data (68% zeros). This is not a bug — it is
  fundamental to recursive autoregression: each wrong prediction contaminates the next
  step's lag features. The gap cannot be eliminated without multi-horizon direct training
  (see Day 9). v2 produces the same gap as v1 because the feature algebra is identical.
"""
from __future__ import annotations

import time
import numpy as np
import pandas as pd

from shelfsense.features.hierarchy import CAT_DTYPES as DEFAULT_CAT_DTYPES
from shelfsense.features.calendar import build_calendar_lookup
from shelfsense.features.price import build_price_lookup

LAGS    = [7, 14, 28, 56]
WINDOWS = [7, 28, 56, 180]

CAL_FEATURE_COLS = [
    "weekday", "month", "quarter", "year", "day_of_month", "week_of_year",
    "is_weekend", "is_holiday", "is_snap_ca", "is_snap_tx", "is_snap_wi",
    "days_since_event", "days_until_next_event",
]
PRICE_FEATURE_COLS = [
    "sell_price", "price_change_pct", "price_relative_mean",
    "price_volatility", "has_price_change",
]
LAG_FEATURES  = [f"lag_{l}" for l in LAGS]
ROLL_FEATURES = [
    f"roll_{stat}_{w}"
    for w in WINDOWS
    for stat in ["mean", "std", "min", "max"]
]
NUM_FEATURES = CAL_FEATURE_COLS + PRICE_FEATURE_COLS + LAG_FEATURES + ROLL_FEATURES
CAT_COLS     = ["cat_id", "dept_id", "store_id", "state_id"]
ALL_FEATURES = NUM_FEATURES + CAT_COLS

HISTORY_DAYS = 200  # days of sales history kept; covers max window (180) + lag_56 margin


def _build_price_by_day(
    series_meta: pd.DataFrame,
    price_lookup: pd.DataFrame,
    cal_lookup: pd.DataFrame,
    start_day: int,
    end_day: int,
) -> dict[int, np.ndarray]:
    """
    Pre-compute (n_series, 5) price feature arrays for every day in [start_day, end_day].
    Groups by wm_yr_wk (at most ~5 unique weeks in a 28-day window).
    Missing item/store combos → 0.
    """
    d_to_wk = {
        d: int(cal_lookup.loc[f"d_{d}", "wm_yr_wk"])
        for d in range(start_day, end_day + 1)
    }
    unique_wks = sorted(set(d_to_wk.values()))

    wk_arrays: dict[int, np.ndarray] = {}
    for wk in unique_wks:
        wk_rows = price_lookup[price_lookup["wm_yr_wk"] == wk].set_index(["item_id", "store_id"])
        merged = (
            series_meta[["item_id", "store_id"]]
            .merge(wk_rows[PRICE_FEATURE_COLS].reset_index(), on=["item_id", "store_id"], how="left")
        )
        wk_arrays[wk] = np.nan_to_num(
            merged[PRICE_FEATURE_COLS].values.astype(np.float32), nan=0.0
        )

    return {d: wk_arrays[d_to_wk[d]] for d in range(start_day, end_day + 1)}


def _build_history_df(
    sales_eval: pd.DataFrame,
    series_order: np.ndarray,
    last_day: int,
    history_days: int = HISTORY_DAYS,
) -> pd.DataFrame:
    """
    Build a long-format history DataFrame from wide-format sales_eval.
    Returns HISTORY_DAYS rows per series, from d_(last_day - history_days + 1) to d_(last_day).
    Pre-launch NaN → 0.

    Parameters
    ----------
    sales_eval   : wide-format evaluation sales (30 490 rows × d_1…d_1941)
    series_order : sorted series IDs to include (and their row ordering in output)
    last_day     : last d_num to include (e.g. 1913 for sanity check, 1941 for eval)
    """
    first_day = last_day - history_days + 1
    day_cols = [f"d_{d}" for d in range(first_day, last_day + 1) if f"d_{d}" in sales_eval.columns]
    meta_cols = ["id", "item_id", "cat_id", "dept_id", "store_id", "state_id"]

    sub = (
        sales_eval[sales_eval["id"].isin(series_order)]
        .set_index("id").reindex(series_order).reset_index()
    )
    df = sub[meta_cols + day_cols].melt(
        id_vars=meta_cols, var_name="d", value_name="sales"
    )
    df["d_num"] = df["d"].str.replace("d_", "", regex=False).astype(np.int32)
    df["sales"] = df["sales"].fillna(0.0).astype(np.float32)
    return df.drop(columns=["d"])


def predict_horizon(
    model,
    history_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    days_out: int = 28,
    cat_dtypes: dict | None = None,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Recursive multi-step forecast starting from the day after history_df's last d_num.

    Parameters
    ----------
    history_df  : long-format DataFrame with columns:
                  [id, item_id, cat_id, dept_id, store_id, state_id, d_num, sales].
                  Must contain at least HISTORY_DAYS (200) consecutive days per series.
    calendar_df : M5 calendar.csv — must cover forecast days.
    prices_df   : M5 sell_prices.csv — must cover forecast-period wm_yr_wk weeks.
    days_out    : number of steps to forecast (28 for M5 eval/val).
    cat_dtypes  : {col: CategoricalDtype} matching model training. Defaults to M5 CAT_DTYPES.

    Returns
    -------
    preds      : (n_series, days_out) float32 — in sorted-id order.
    series_ids : (n_series,) ndarray of series IDs matching preds rows.
    """
    if cat_dtypes is None:
        cat_dtypes = DEFAULT_CAT_DTYPES

    # Lookups (cheap; built once)
    cal_lookup   = build_calendar_lookup(calendar_df)
    price_lookup = build_price_lookup(prices_df, calendar_df)

    series_meta = (
        history_df[["id", "item_id", "cat_id", "dept_id", "store_id", "state_id"]]
        .drop_duplicates("id")
        .sort_values("id")
        .reset_index(drop=True)
    )
    series_ids = series_meta["id"].values
    n_series   = len(series_ids)

    last_d    = int(history_df["d_num"].max())
    start_day = last_d + 1
    end_day   = last_d + days_out

    # Pre-compute price arrays for all forecast days (grouped by week → ~5 unique lookups)
    price_by_d = _build_price_by_day(
        series_meta, price_lookup, cal_lookup,
        start_day=start_day, end_day=end_day,
    )

    # Sales matrix: (n_series, n_hist) — pivot history to 2-D array
    recent = history_df[history_df["d_num"] >= (last_d - HISTORY_DAYS + 1)]
    sales_wide = (
        recent.pivot_table(index="id", columns="d_num", values="sales", aggfunc="first")
        .reindex(series_ids)
        .fillna(0.0)
        .astype(np.float32)
    )
    day_cols  = np.array(sorted(sales_wide.columns), dtype=np.int32)   # (n_hist,)
    sales_mat = sales_wide[day_cols].values                             # (n_series, n_hist)

    # Cat feature arrays (constant across steps)
    cat_arrs = {
        col: series_meta[col].astype(str).values for col in CAT_COLS
    }

    preds   = np.zeros((n_series, days_out), dtype=np.float32)
    t_total = time.time()

    for step in range(days_out):
        t0 = time.time()
        d  = start_day + step

        # Lag features: exact day lookup via searchsorted
        lag_list = []
        for lag in LAGS:
            target_d = d - lag
            pos = int(np.searchsorted(day_cols, target_d))
            if pos < len(day_cols) and day_cols[pos] == target_d:
                lag_list.append(sales_mat[:, pos])
            else:
                lag_list.append(np.zeros(n_series, dtype=np.float32))
        lag_mat = np.column_stack(lag_list)   # (n_series, 4)

        # Rolling features: mask over days in [d-w, d-1]
        roll_list = []
        for w in WINDOWS:
            d_lo, d_hi = d - w, d - 1
            in_win = (day_cols >= d_lo) & (day_cols <= d_hi)
            if not in_win.any():
                roll_list += [np.zeros(n_series, np.float32)] * 4
                continue
            win = sales_mat[:, in_win]                      # (n_series, k)
            roll_list.append(win.mean(axis=1).astype(np.float32))
            if win.shape[1] > 1:
                std_ = np.nan_to_num(win.std(axis=1, ddof=1), nan=0.0).astype(np.float32)
            else:
                std_ = np.zeros(n_series, np.float32)
            roll_list.append(std_)
            roll_list.append(win.min(axis=1).astype(np.float32))
            roll_list.append(win.max(axis=1).astype(np.float32))
        roll_mat = np.column_stack(roll_list)  # (n_series, 16)

        # Calendar (broadcast)
        cal_row = cal_lookup.loc[f"d_{d}", CAL_FEATURE_COLS].values.astype(np.float32)
        cal_mat = np.tile(cal_row, (n_series, 1))   # (n_series, 13)

        # Price (pre-computed per day)
        price_mat = price_by_d[d]                   # (n_series, 5)

        # Assemble feature DataFrame
        num_mat = np.hstack([cal_mat, price_mat, lag_mat, roll_mat]).astype(np.float32)
        feat_df = pd.DataFrame(num_mat, columns=NUM_FEATURES)
        for col in CAT_COLS:
            feat_df[col] = pd.Categorical(
                cat_arrs[col],
                categories=cat_dtypes[col].categories,
                ordered=False,
            )
        feat_df = feat_df[ALL_FEATURES]

        # Predict + clip
        step_preds = np.clip(model.predict(feat_df), 0.0, None).astype(np.float32)
        preds[:, step] = step_preds

        # Extend sales matrix with this step's predictions
        sales_mat = np.hstack([sales_mat, step_preds.reshape(-1, 1)])
        day_cols  = np.append(day_cols, d)

        if verbose:
            print(f"    d_{d}  ({time.time() - t0:.1f}s)", end="  ", flush=True)
            if (step + 1) % 7 == 0 or step == days_out - 1:
                print(flush=True)

    if verbose:
        print(f"    Total: {time.time() - t_total:.1f}s")

    return preds, series_ids
