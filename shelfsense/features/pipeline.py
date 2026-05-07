"""
Feature engineering pipeline for M5.

Public API:
    feature_engineer(sales_df, calendar_df, prices_df, output_dir, last_day=1941)
        -> int  (writes partitioned parquet to output_dir; returns total rows written)
    feature_engineer_from_config(cfg, output_dir=None)
        -> int  (loads raw CSVs from cfg.data.raw_dir, writes to cfg.data.processed_dir)

Processing strategy:
  - Batch by store_id (10 stores, ~5.9M rows each) to keep peak memory <= ~1.5 GB per batch.
  - Each store is written as an independent parquet file.

Output schema (per row = one series x one day):
  Meta    : id, item_id, dept_id, cat_id, store_id, state_id (cat dtype)
  Target  : sales (float32)
  Index   : d (str), d_num (int16)
  Calendar: weekday, month, quarter, year, day_of_month, week_of_year,
            is_weekend, is_holiday, is_snap_ca/tx/wi,
            days_since_event, days_until_next_event
  Price   : sell_price, price_change_pct, price_relative_mean,
            price_volatility, has_price_change
  Lags    : lag_7, lag_14, lag_28, lag_56, lag_91, lag_182, lag_364
  Rolling : roll_{mean,std,min,max}_{7,28,56,180}  (16 cols)
"""
from __future__ import annotations

import os
import time
import warnings

import numpy as np
import pandas as pd

from shelfsense.features.lags import add_lags
from shelfsense.features.rolling import add_rolling
from shelfsense.features.calendar import build_calendar_lookup, add_calendar_features
from shelfsense.features.price import build_price_lookup, add_price_features
from shelfsense.features.hierarchy import add_hierarchy_features

warnings.filterwarnings("ignore")

META_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]


def feature_engineer(
    sales_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    output_dir: str,
    last_day: int = 1941,
    verbose: bool = True,
) -> int:
    """
    Build and write features for all 30,490 series x last_day days.

    Parameters
    ----------
    sales_df :
        Wide-format evaluation sales (30,490 rows, d_1 .. d_last_day columns).
    calendar_df :
        M5 calendar CSV as a DataFrame.
    prices_df :
        M5 sell_prices CSV as a DataFrame.
    output_dir :
        Directory to write partitioned parquet files (one per store).
    last_day :
        Last day column to include. Default 1941 = full evaluation file.
    verbose :
        Print per-store progress.

    Returns
    -------
    int
        Total number of rows written across all stores.
    """
    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print("  Building calendar lookup...", flush=True)
    cal_lookup = build_calendar_lookup(calendar_df)

    if verbose:
        print("  Building price lookup...", flush=True)
    price_lookup = build_price_lookup(prices_df, calendar_df)

    day_cols = [f"d_{d}" for d in range(1, last_day + 1) if f"d_{d}" in sales_df.columns]

    stores = sorted(sales_df["store_id"].unique())
    total_rows = 0
    t_total = time.time()

    for store in stores:
        out_path = os.path.join(output_dir, f"store_{store}.parquet")
        if os.path.exists(out_path):
            size_mb = os.path.getsize(out_path) / 1e6
            if verbose:
                print(f"  [{store}] Already exists ({size_mb:.1f} MB), skipping.", flush=True)
            import pyarrow.parquet as pq
            meta = pq.read_metadata(out_path)
            total_rows += meta.num_rows
            continue

        t_store = time.time()
        store_mask = sales_df["store_id"] == store
        store_sales = sales_df[store_mask].copy()
        n_series = len(store_sales)

        if verbose:
            print(f"  [{store}] {n_series} series, {len(day_cols)} days...", flush=True)

        df = store_sales[META_COLS + day_cols].melt(
            id_vars=META_COLS,
            var_name="d",
            value_name="sales",
        )
        df["sales"] = df["sales"].astype(np.float32)

        df = add_calendar_features(df, cal_lookup)
        df = add_price_features(df, price_lookup)
        df = df.sort_values(["id", "d_num"]).reset_index(drop=True)
        df = add_lags(df)
        df = add_rolling(df)
        df = add_hierarchy_features(df)
        df = df.drop(columns=["wm_yr_wk"], errors="ignore")

        if "has_price_change" in df.columns:
            df["has_price_change"] = df["has_price_change"].fillna(0).astype(np.int8)
        if "sell_price" in df.columns:
            df["sell_price"] = df["sell_price"].fillna(0.0).astype(np.float32)

        df["d_num"] = df["d_num"].astype(np.int16)

        df.to_parquet(out_path, index=False, compression="snappy")

        n_rows = len(df)
        total_rows += n_rows
        elapsed = time.time() - t_store
        if verbose:
            size_mb = os.path.getsize(out_path) / 1e6
            print(f"    Wrote {n_rows:,} rows -> {out_path} ({size_mb:.1f} MB) in {elapsed:.1f}s", flush=True)

        del df, store_sales

    total_elapsed = time.time() - t_total
    if verbose:
        print(f"\n  Done. Total rows: {total_rows:,}  Time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")

    return total_rows

def feature_engineer_from_config(cfg, output_dir: str | None = None) -> int:
    """
    Load raw M5 CSVs and call feature_engineer() using paths from a Hydra config.

    Parameters
    ----------
    cfg : OmegaConf DictConfig
        Must have cfg.data.raw_dir, cfg.data.processed_dir,
        cfg.data.last_train_day, and cfg.data.horizon.
    output_dir :
        Override for the output directory. When None, uses cfg.data.processed_dir.

    Returns
    -------
    int
        Total rows written (forwarded from feature_engineer).
    """
    raw_dir = cfg.data.raw_dir
    out_dir = output_dir if output_dir is not None else cfg.data.processed_dir
    last_day = cfg.data.last_train_day + cfg.data.horizon  # default: 1913 + 28 = 1941

    print(f"Loading raw CSVs from {raw_dir!r}...", flush=True)
    sales_df = pd.read_csv(os.path.join(raw_dir, "sales_train_evaluation.csv"))
    calendar_df = pd.read_csv(os.path.join(raw_dir, "calendar.csv"))
    prices_df = pd.read_csv(os.path.join(raw_dir, "sell_prices.csv"))

    print(f"Writing features to {out_dir!r}  (last_day={last_day})", flush=True)
    return feature_engineer(sales_df, calendar_df, prices_df, out_dir, last_day=last_day)
