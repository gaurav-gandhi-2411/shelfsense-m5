"""
Price feature builder for M5.

build_price_lookup(prices_df, calendar_df) -> pd.DataFrame indexed by (item_id, store_id, wm_yr_wk)
    Computes per-series, per-week price features.

add_price_features(df, price_lookup) -> df
    Joins price features onto long-format sales DataFrame via (item_id, store_id, wm_yr_wk).
    wm_yr_wk must already be present in df (added by add_calendar_features).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_price_lookup(
    prices_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute price features at (item_id, store_id, wm_yr_wk) granularity.

    Features:
      sell_price          – raw sell price (float32)
      price_change_pct    – (price_t - price_{t-1}) / price_{t-1}; NaN for first week
      price_relative_mean – sell_price / item_store's all-time mean price
      price_volatility    – rolling std of weekly price over last 28 weeks (min 2)
      has_price_change    – 1 if |price_change_pct| > 1% (non-trivial change)
    """
    p = prices_df.copy().sort_values(["store_id", "item_id", "wm_yr_wk"])

    # week-over-week price change
    p["price_prev"] = p.groupby(["store_id", "item_id"])["sell_price"].shift(1)
    p["price_change_pct"] = (
        (p["sell_price"] - p["price_prev"]) / p["price_prev"]
    ).astype(np.float32)
    p["has_price_change"] = (p["price_change_pct"].abs() > 0.01).astype(np.int8)

    # price relative to item-store mean (computed over all available weeks)
    p["price_relative_mean"] = (
        p["sell_price"]
        / p.groupby(["store_id", "item_id"])["sell_price"].transform("mean")
    ).astype(np.float32)

    # price volatility: rolling std over 28 weeks (shift 1 to avoid same-week leakage)
    p["price_volatility"] = (
        p.groupby(["store_id", "item_id"])["sell_price"]
        .transform(lambda x: x.shift(1).rolling(28, min_periods=2).std())
    ).astype(np.float32)

    p["sell_price"] = p["sell_price"].astype(np.float32)

    keep = [
        "store_id", "item_id", "wm_yr_wk",
        "sell_price", "price_change_pct", "price_relative_mean",
        "price_volatility", "has_price_change",
    ]
    return p[keep]


def add_price_features(df: pd.DataFrame, price_lookup: pd.DataFrame) -> pd.DataFrame:
    """
    Merge price features onto df via (item_id, store_id, wm_yr_wk).
    wm_yr_wk must already be in df (from calendar join).
    """
    return df.merge(
        price_lookup,
        on=["item_id", "store_id", "wm_yr_wk"],
        how="left",
    )
