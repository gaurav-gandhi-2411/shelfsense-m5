"""Synthetic M5-shaped datasets for fast, data-free tests.

Public API
----------
make_sales_df(n_days, stores, n_items_per_store)
    Wide-format sales_train_evaluation.csv replica (id + d_1..d_N cols).

make_calendar_df(n_days)
    calendar.csv replica with SNAP flags and a few events.

make_prices_df(sales_df, calendar_df)
    sell_prices.csv replica.

write_synthetic_csvs(raw_dir, n_days, n_items_per_store)
    Write all three CSVs to raw_dir; return raw_dir.

make_features_df(n_days, n_series)
    Run the full feature pipeline on synthetic data and return the
    concatenated long-format DataFrame (for unit tests that need feature rows).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

STORES = ["CA_1", "CA_2", "CA_3", "CA_4", "TX_1", "TX_2", "TX_3", "WI_1", "WI_2", "WI_3"]
DEPTS  = ["FOODS_1", "FOODS_2", "FOODS_3", "HOBBIES_1", "HOBBIES_2", "HOUSEHOLD_1", "HOUSEHOLD_2"]
CATS   = {
    "FOODS_1": "FOODS",    "FOODS_2": "FOODS",    "FOODS_3": "FOODS",
    "HOBBIES_1": "HOBBIES","HOBBIES_2": "HOBBIES",
    "HOUSEHOLD_1": "HOUSEHOLD", "HOUSEHOLD_2": "HOUSEHOLD",
}
STATES = {
    "CA_1": "CA", "CA_2": "CA", "CA_3": "CA", "CA_4": "CA",
    "TX_1": "TX", "TX_2": "TX", "TX_3": "TX",
    "WI_1": "WI", "WI_2": "WI", "WI_3": "WI",
}


def make_sales_df(
    n_days: int = 400,
    stores: list[str] | None = None,
    n_items_per_store: int = 10,
    seed: int = 0,
) -> pd.DataFrame:
    """Return a wide-format sales DataFrame matching M5's sales_train_evaluation schema."""
    rng = np.random.default_rng(seed)
    _stores = stores or STORES
    rows = []
    for store in _stores:
        state = STATES[store]
        for i in range(n_items_per_store):
            dept = DEPTS[i % len(DEPTS)]
            cat  = CATS[dept]
            item_id = f"{dept}_{i:03d}"
            row = {
                "id":       f"{item_id}_{store}_evaluation",
                "item_id":  item_id,
                "dept_id":  dept,
                "cat_id":   cat,
                "store_id": store,
                "state_id": state,
            }
            row.update({f"d_{d}": int(rng.poisson(2)) for d in range(1, n_days + 1)})
            rows.append(row)
    return pd.DataFrame(rows)


def make_calendar_df(n_days: int = 400) -> pd.DataFrame:
    """Return a calendar DataFrame matching M5's calendar.csv schema."""
    base = pd.Timestamp("2011-01-29")
    rows = []
    for d in range(1, n_days + 1):
        dt = base + pd.Timedelta(days=d - 1)
        wm_yr_wk = int(dt.strftime("%Y%V"))
        event1 = "NewYear" if d == 1 else ("Christmas" if d == 359 else None)
        rows.append({
            "d":            f"d_{d}",
            "date":         dt.strftime("%Y-%m-%d"),
            "wm_yr_wk":     wm_yr_wk,
            "event_name_1": event1,
            "event_name_2": None,
            "snap_CA":      int(d % 7 == 0),
            "snap_TX":      int(d % 7 == 1),
            "snap_WI":      int(d % 7 == 2),
        })
    return pd.DataFrame(rows)


def make_prices_df(sales_df: pd.DataFrame, calendar_df: pd.DataFrame, seed: int = 1) -> pd.DataFrame:
    """Return a sell_prices DataFrame for the series in sales_df."""
    rng = np.random.default_rng(seed)
    wm_yr_wks = sorted(calendar_df["wm_yr_wk"].unique())
    items = sales_df[["item_id", "store_id"]].drop_duplicates()
    rows = []
    for _, row in items.iterrows():
        for wk in wm_yr_wks:
            rows.append({
                "store_id":   row["store_id"],
                "item_id":    row["item_id"],
                "wm_yr_wk":   wk,
                "sell_price": round(float(rng.uniform(1.0, 10.0)), 2),
            })
    return pd.DataFrame(rows)


def write_synthetic_csvs(
    raw_dir: str,
    n_days: int = 400,
    n_items_per_store: int = 10,
    stores: list[str] | None = None,
) -> str:
    """Write synthetic CSVs to raw_dir and return raw_dir."""
    os.makedirs(raw_dir, exist_ok=True)
    sales_df  = make_sales_df(n_days=n_days, stores=stores, n_items_per_store=n_items_per_store)
    cal_df    = make_calendar_df(n_days=n_days)
    prices_df = make_prices_df(sales_df, cal_df)
    sales_df.to_csv(os.path.join(raw_dir, "sales_train_evaluation.csv"),  index=False)
    cal_df.to_csv(  os.path.join(raw_dir, "calendar.csv"),                index=False)
    prices_df.to_csv(os.path.join(raw_dir, "sell_prices.csv"),            index=False)
    return raw_dir


def make_features_df(n_days: int = 400, n_items_per_store: int = 1) -> pd.DataFrame:
    """Run the feature pipeline on synthetic data and return the long-format DataFrame.

    Useful for unit tests that need fully-formed feature rows without writing to disk.
    Each store gets one parquet; this function concatenates them.
    """
    import tempfile

    from omegaconf import OmegaConf

    from shelfsense.features.pipeline import feature_engineer_from_config

    sales_df  = make_sales_df(n_days=n_days, n_items_per_store=n_items_per_store)
    cal_df    = make_calendar_df(n_days=n_days)
    prices_df = make_prices_df(sales_df, cal_df)

    with tempfile.TemporaryDirectory() as raw_dir, tempfile.TemporaryDirectory() as out_dir:
        sales_df.to_csv(  os.path.join(raw_dir, "sales_train_evaluation.csv"), index=False)
        cal_df.to_csv(    os.path.join(raw_dir, "calendar.csv"),               index=False)
        prices_df.to_csv( os.path.join(raw_dir, "sell_prices.csv"),            index=False)

        cfg = OmegaConf.create({
            "data": {
                "raw_dir":        raw_dir,
                "processed_dir":  out_dir,
                "last_train_day": n_days - 28,
                "horizon":        28,
            }
        })
        feature_engineer_from_config(cfg, output_dir=out_dir)

        frames = []
        for fname in os.listdir(out_dir):
            if fname.endswith(".parquet"):
                frames.append(pd.read_parquet(os.path.join(out_dir, fname)))

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
