"""
WRMSSE (Weighted Root Mean Squared Scaled Error) evaluator for M5.

  WRMSSE = (1/12) * sum_{l=1}^{12}  sum_{i in level_l}  w_{i,l} * RMSSE_{i,l}

  RMSSE_{i,l}  = sqrt( MSE(forecast_i, actual_i) / scale_{i,l} )
  scale_{i,l}  = mean( (y_t - y_{t-1})^2 )  over full training history of
                 the aggregated series i at level l
  w_{i,l}      = dollar revenue of series i in last 28 training days,
                 normalised to sum=1 within level l
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

# ── 12 M5 hierarchical levels ────────────────────────────────────────────────
# Value = list of meta-columns to group by; [] means grand total (level 1).
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
    "level_11": ["state_id", "item_id"],
    "level_12": ["store_id", "item_id"],   # base: 30 490 series
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
    mat: np.ndarray,      # (n_series, T)
    keys: np.ndarray,     # (n_series,)
) -> Tuple[np.ndarray, np.ndarray]:
    """Sum rows of *mat* by *keys*.  Returns (agg_mat, unique_keys)."""
    ukeys, inv = np.unique(keys, return_inverse=True)
    agg = np.zeros((len(ukeys), mat.shape[1]), dtype=np.float64)
    np.add.at(agg, inv, mat.astype(np.float64))
    return agg, ukeys


# ── public building blocks ────────────────────────────────────────────────────

def build_scales(train_mat: np.ndarray) -> np.ndarray:
    """
    Naive-1 MSE scale per series.

    scale_i = mean( (y_t - y_{t-1})^2 )  over all training history.
    Constant series (scale=0) are set to 1.0 so RMSSE = sqrt(MSE).
    """
    diff = np.diff(train_mat.astype(np.float64), axis=1)
    scales = np.mean(diff ** 2, axis=1)
    return np.where(scales == 0, 1.0, scales)


def build_revenue_weights(
    sales_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    last_train_day: int = 1913,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Revenue weights for all 12 levels.

    Returns {level: (sorted_group_keys, weights)}, weights sum to 1 per level.
    """
    start_day = last_train_day - HORIZON + 1
    day_cols = [c for c in (f"d_{d}" for d in range(start_day, last_train_day + 1))
                if c in sales_df.columns]

    weeks = calendar_df[calendar_df["d"].isin(day_cols)]["wm_yr_wk"].unique()
    avg_price = (
        prices_df[prices_df["wm_yr_wk"].isin(weeks)]
        .groupby(["store_id", "item_id"])["sell_price"]
        .mean()
    )

    meta = sales_df[["id", "store_id", "item_id"]].copy()
    meta = meta.merge(
        avg_price.reset_index().rename(columns={"sell_price": "avg_price"}),
        on=["store_id", "item_id"],
        how="left",
    )
    meta["avg_price"] = meta["avg_price"].fillna(0.0)

    sales_28 = sales_df[day_cols].values.sum(axis=1)
    revenue = sales_28 * meta["avg_price"].values          # (30 490,)

    meta_full = sales_df[META_COLS].copy()
    revenue_col = revenue.reshape(-1, 1).astype(np.float64)

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
    scales: np.ndarray,   # (30 490,)
) -> np.ndarray:
    """RMSSE for each of the 30 490 base series."""
    mse = np.mean((preds.astype(np.float64) - actuals.astype(np.float64)) ** 2, axis=1)
    return np.sqrt(mse / scales)


def compute_wrmsse(
    preds: np.ndarray,          # (30 490, 28) — row-aligned with sales_df
    actuals: np.ndarray,        # (30 490, 28)
    sales_df: pd.DataFrame,     # full training data (sales_train_validation or _evaluation)
    prices_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    last_train_day: int = 1913,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute M5 WRMSSE across all 12 hierarchical levels.

    Returns (total_wrmsse, {level_name: level_wrmsse}).
    total_wrmsse = unweighted mean of 12 level scores.
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
        rmsse = np.sqrt(mse / level_scales)

        ukeys_w, w_vals = weights[level]
        key_to_w = dict(zip(ukeys_w, w_vals))
        w = np.array([key_to_w.get(k, 0.0) for k in ukeys])
        ws = w.sum()
        if ws > 0:
            w /= ws

        level_scores[level] = float(np.dot(w, rmsse))

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
