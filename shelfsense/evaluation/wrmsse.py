"""
WRMSSE (Weighted Root Mean Squared Scaled Error) evaluator for M5.

  WRMSSE = (1/12) * sum_{l=1}^{12}  sum_{i in level_l}  w_{i,l} * RMSSE_{i,l}

  RMSSE_{i,l}  = sqrt( MSE(forecast_i, actual_i) / scale_{i,l} )
  scale_{i,l}  = mean( (y_t - y_{t-1})^2 )  computed from the FIRST NON-ZERO
                 observation onward in the training history of aggregated series i
                 (matches the reference WRMSSEEvaluator from the A4 winning solution)
  w_{i,l}      = dollar revenue of series i in last 28 training days (day-level
                 sales × week price), normalised to sum=1 within level l

Key fix vs original implementation:
- build_scales now trims leading zeros before computing the naive-1 MSE denominator.
  Many M5 series launch partway through the dataset; including pre-launch zeros
  deflates the scale and artificially inflates RMSSE.
- build_revenue_weights now uses per-day price joins (sales_d × price_week(d))
  instead of avg_price × total_sales_28.
- Series with scale=0 (all-zero training history) produce RMSSE=inf and are
  excluded from the weighted sum (weight contribution dropped), matching the
  reference implementation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

# ── 12 M5 hierarchical levels ────────────────────────────────────────────────
LEVEL_SPECS: Dict[str, List[str]] = {
    "level_1":  [],
    "level_2":  ["state_id"],
    "level_3":  ["store_id"],
    "level_4":  ["cat_id"],
    "level_5":  ["dept_id"],
    "level_6":  ["state_id", "cat_id"],
    "level_7":  ["state_id", "dept_id"],
    "level_8":  ["store_id", "cat_id"],
    "level_9":  ["store_id", "dept_id"],
    "level_10": ["item_id"],
    "level_11": ["item_id", "state_id"],
    "level_12": ["item_id", "store_id"],   # base: 30 490 series
}

META_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
HORIZON = 28


# ── internal helpers ─────────────────────────────────────────────────────────

def _group_keys(meta: pd.DataFrame, cols: List[str]) -> np.ndarray:
    if not cols:
        return np.full(len(meta), "total")
    if len(cols) == 1:
        return meta[cols[0]].values.astype(str)
    return (
        meta[cols]
        .apply(lambda r: "__".join(r.values.astype(str)), axis=1)
        .values
    )


def _aggregate(
    mat: np.ndarray,   # (n_series, T)
    keys: np.ndarray,  # (n_series,)
) -> Tuple[np.ndarray, np.ndarray]:
    """Sum rows of *mat* by *keys*.  Returns (agg_mat, unique_keys)."""
    ukeys, inv = np.unique(keys, return_inverse=True)
    agg = np.zeros((len(ukeys), mat.shape[1]), dtype=np.float64)
    np.add.at(agg, inv, mat.astype(np.float64))
    return agg, ukeys


# ── public building blocks ────────────────────────────────────────────────────

def build_scales(train_mat: np.ndarray) -> np.ndarray:
    """
    Naive-1 MSE scale per series, trimming leading zeros.

    For each series: find the first non-zero day, then compute
      scale = mean( (y_t - y_{t-1})^2 )
    over that active window.  All-zero series get scale=0 (their RMSSE
    becomes inf and is excluded from the weighted average downstream).
    """
    n = train_mat.shape[0]
    scales = np.zeros(n, dtype=np.float64)
    mat = train_mat.astype(np.float64)

    for i in range(n):
        row = mat[i]
        first_nz = int(np.argmax(row != 0))
        # argmax returns 0 for all-False; distinguish true first-nz from all-zeros
        if row[first_nz] == 0:          # all-zero series
            scales[i] = 0.0
            continue
        active = row[first_nz:]
        if len(active) < 2:
            scales[i] = 1.0             # can't compute diff — use neutral scale
        else:
            d = np.diff(active)
            scales[i] = float(np.mean(d ** 2))

    return scales


def build_revenue_weights(
    sales_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    last_train_day: int = 1913,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Revenue weights for all 12 levels.

    Revenue = sum over last 28 training days of (sales_day × price_week_of_day).
    Each day is matched to its exact wm_yr_wk before joining prices.

    Returns {level: (sorted_group_keys, weights)}, weights sum to 1 per level.
    """
    start_day = last_train_day - HORIZON + 1
    day_cols = [c for c in (f"d_{d}" for d in range(start_day, last_train_day + 1))
                if c in sales_df.columns]

    day_to_week = calendar_df.set_index("d")["wm_yr_wk"].to_dict()

    # Melt to long format for exact day-level price join (matches reference)
    weight_long = (
        sales_df[["item_id", "store_id"] + day_cols]
        .melt(id_vars=["item_id", "store_id"], var_name="d", value_name="sales")
    )
    weight_long["wm_yr_wk"] = weight_long["d"].map(day_to_week)
    weight_long = weight_long.merge(
        prices_df[["item_id", "store_id", "wm_yr_wk", "sell_price"]],
        on=["item_id", "store_id", "wm_yr_wk"],
        how="left",
    )
    weight_long["sell_price"] = weight_long["sell_price"].fillna(0.0)
    weight_long["revenue"] = weight_long["sales"] * weight_long["sell_price"]

    # Aggregate back to series level in sales_df row order
    rev_by_series = (
        weight_long.groupby(["item_id", "store_id"])["revenue"]
        .sum()
        .reset_index()
    )
    # META_COLS already contains item_id and store_id — no duplicate needed
    meta = sales_df[META_COLS].copy()
    meta = meta.merge(rev_by_series, on=["item_id", "store_id"], how="left")
    revenue = meta["revenue"].fillna(0.0).values  # (30 490,)

    revenue_col = revenue.reshape(-1, 1).astype(np.float64)
    meta_full = sales_df[META_COLS].copy()

    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for level, cols in LEVEL_SPECS.items():
        keys = _group_keys(meta_full, cols)
        rev_agg, ukeys = _aggregate(revenue_col, keys)
        rev_agg = rev_agg[:, 0]
        total = rev_agg.sum()
        w = rev_agg / total if total > 0 else np.ones(len(rev_agg)) / len(rev_agg)
        out[level] = (ukeys, w)

    return out


def compute_rmsse_per_series(
    preds: np.ndarray,    # (30 490, 28)
    actuals: np.ndarray,  # (30 490, 28)
    scales: np.ndarray,   # (30 490,)  — from build_scales
) -> np.ndarray:
    """RMSSE for each of the 30 490 base series (may contain inf for scale=0)."""
    mse = np.mean((preds.astype(np.float64) - actuals.astype(np.float64)) ** 2, axis=1)
    return np.sqrt(mse / scales)   # scale=0 → inf, handled by caller


def compute_wrmsse(
    preds: np.ndarray,          # (30 490, 28) — row-aligned with sales_df
    actuals: np.ndarray,        # (30 490, 28)
    sales_df: pd.DataFrame,     # training data (sales_train_validation or _evaluation)
    prices_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    last_train_day: int = 1913,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute M5 WRMSSE across all 12 hierarchical levels.

    Returns (total_wrmsse, {level_name: level_wrmsse}).
    total_wrmsse = unweighted mean of 12 level scores.
    Infinite RMSSE terms (all-zero series) are excluded from the weighted sum.
    """
    meta = sales_df[META_COLS].copy()

    train_cols = [c for c in (f"d_{d}" for d in range(1, last_train_day + 1))
                  if c in sales_df.columns]
    train_mat = sales_df[train_cols].values.astype(np.float64)

    weights = build_revenue_weights(sales_df, prices_df, calendar_df, last_train_day)

    level_scores: Dict[str, float] = {}

    for level, cols in LEVEL_SPECS.items():
        keys = _group_keys(meta, cols)

        pred_agg,  ukeys = _aggregate(preds.astype(np.float64), keys)
        act_agg,   _     = _aggregate(actuals.astype(np.float64), keys)
        train_agg, _     = _aggregate(train_mat, keys)

        level_scales = build_scales(train_agg)
        mse = np.mean((pred_agg - act_agg) ** 2, axis=1)
        rmsse = np.sqrt(mse / level_scales)   # may contain inf

        ukeys_w, w_vals = weights[level]
        key_to_w = dict(zip(ukeys_w, w_vals))
        w = np.array([key_to_w.get(k, 0.0) for k in ukeys])
        ws = w.sum()
        if ws > 0:
            w /= ws

        # Exclude inf RMSSE (scale=0 series) — matches reference implementation
        finite = np.isfinite(rmsse)
        score = float(np.sum(w[finite] * rmsse[finite]))
        level_scores[level] = score

    total = float(np.mean(list(level_scores.values())))
    return total, level_scores


def submission_to_matrix(
    sub_df: pd.DataFrame,    # Kaggle-format: id + F1..F28
    sales_df: pd.DataFrame,  # to get the canonical row ordering
    suffix: str = "_validation",
) -> np.ndarray:
    """
    Align a Kaggle submission DataFrame to a (30 490, 28) array in sales_df row order.
    """
    fcols = [f"F{i}" for i in range(1, HORIZON + 1)]
    val_rows = (
        sub_df[sub_df["id"].str.endswith(suffix)]
        .set_index("id")
    )
    result = val_rows.reindex(sales_df["id"].values)[fcols].fillna(0.0).values
    return result.astype(np.float32)
