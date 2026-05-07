"""
Integration test for feature_engineer_from_config.

Uses a synthetic 10-series × 400-day fixture so the test runs without M5 data.
The fixture mirrors the production schema: same column names, same dtypes, same
number of stores (CA_1..WI_3), but only 10 items per store instead of 3,049.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from shelfsense.features.pipeline import feature_engineer_from_config


N_DAYS = 400        # enough for all 7 lag windows (max=364) plus a margin
N_ITEMS_PER_STORE = 2
STORES = ["CA_1", "CA_2", "CA_3", "CA_4", "TX_1", "TX_2", "TX_3", "WI_1", "WI_2", "WI_3"]
DEPTS = ["FOODS_1", "FOODS_2", "FOODS_3", "HOBBIES_1", "HOBBIES_2", "HOUSEHOLD_1", "HOUSEHOLD_2"]
CATS = {"FOODS_1": "FOODS", "FOODS_2": "FOODS", "FOODS_3": "FOODS",
        "HOBBIES_1": "HOBBIES", "HOBBIES_2": "HOBBIES",
        "HOUSEHOLD_1": "HOUSEHOLD", "HOUSEHOLD_2": "HOUSEHOLD"}
STATES = {"CA_1": "CA", "CA_2": "CA", "CA_3": "CA", "CA_4": "CA",
          "TX_1": "TX", "TX_2": "TX", "TX_3": "TX",
          "WI_1": "WI", "WI_2": "WI", "WI_3": "WI"}


def _make_sales(n_days: int = N_DAYS) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for store in STORES:
        state = STATES[store]
        for i in range(N_ITEMS_PER_STORE):
            dept = DEPTS[i % len(DEPTS)]
            cat = CATS[dept]
            item_id = f"{dept}_{i:03d}"
            row = {
                "id": f"{item_id}_{store}_evaluation",
                "item_id": item_id,
                "dept_id": dept,
                "cat_id": cat,
                "store_id": store,
                "state_id": state,
            }
            row.update({f"d_{d}": int(rng.poisson(3)) for d in range(1, n_days + 1)})
            rows.append(row)
    return pd.DataFrame(rows)


def _make_calendar(n_days: int = N_DAYS) -> pd.DataFrame:
    base = pd.Timestamp("2011-01-29")
    rows = []
    for d in range(1, n_days + 1):
        dt = base + pd.Timedelta(days=d - 1)
        wm_yr_wk = int(dt.strftime("%Y%V"))
        # Sprinkle events so calendar.py's event_days array is non-empty
        event1 = "NewYear" if d == 1 else ("Christmas" if d == 359 else None)
        rows.append({
            "d": f"d_{d}",
            "date": dt.strftime("%Y-%m-%d"),
            "wm_yr_wk": wm_yr_wk,
            "event_name_1": event1,
            "event_name_2": None,
            "snap_CA": int(d % 7 == 0),
            "snap_TX": int(d % 7 == 1),
            "snap_WI": int(d % 7 == 2),
        })
    return pd.DataFrame(rows)


def _make_prices(sales_df: pd.DataFrame, calendar_df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    rows = []
    wm_yr_wks = calendar_df["wm_yr_wk"].unique()
    items = sales_df[["item_id", "store_id"]].drop_duplicates()
    for _, row in items.iterrows():
        for wk in wm_yr_wks:
            rows.append({
                "store_id": row["store_id"],
                "item_id": row["item_id"],
                "wm_yr_wk": wk,
                "sell_price": round(float(rng.uniform(1.0, 10.0)), 2),
            })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def synthetic_cfg(tmp_path_factory):
    """Write synthetic CSVs to a temp raw_dir, return a minimal OmegaConf config."""
    raw_dir = str(tmp_path_factory.mktemp("raw") / "m5-forecasting-accuracy")
    os.makedirs(raw_dir)

    sales_df = _make_sales()
    cal_df = _make_calendar()
    prices_df = _make_prices(sales_df, cal_df)

    sales_df.to_csv(os.path.join(raw_dir, "sales_train_evaluation.csv"), index=False)
    cal_df.to_csv(os.path.join(raw_dir, "calendar.csv"), index=False)
    prices_df.to_csv(os.path.join(raw_dir, "sell_prices.csv"), index=False)

    cfg = OmegaConf.create({
        "data": {
            "raw_dir": raw_dir,
            "processed_dir": "UNUSED",   # overridden by output_dir in the test
            "last_train_day": N_DAYS - 28,
            "horizon": 28,
        }
    })
    return cfg


def test_feature_engineer_from_config_runs(synthetic_cfg, tmp_path):
    """Pipeline runs end-to-end and writes one parquet per store."""
    out_dir = str(tmp_path / "features")
    n_rows = feature_engineer_from_config(synthetic_cfg, output_dir=out_dir)

    written = sorted(os.listdir(out_dir))
    assert written == [f"store_{s}.parquet" for s in sorted(STORES)], \
        f"Expected 10 parquets, got: {written}"
    assert n_rows > 0


def test_output_schema(synthetic_cfg, tmp_path):
    """Written parquet contains the required columns and dtypes."""
    out_dir = str(tmp_path / "features")
    feature_engineer_from_config(synthetic_cfg, output_dir=out_dir)

    df = pd.read_parquet(os.path.join(out_dir, "store_CA_1.parquet"))

    required_cols = [
        "id", "item_id", "dept_id", "cat_id", "store_id", "state_id",
        "sales", "d", "d_num",
        "lag_7", "lag_14", "lag_28", "lag_56", "lag_91", "lag_182", "lag_364",
        "roll_mean_7", "roll_mean_28",
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"

    assert df["d_num"].dtype == np.int16
    assert df["sales"].dtype == np.float32


def test_idempotent(synthetic_cfg, tmp_path):
    """Running twice with the same output dir returns the same row count (skip-existing logic)."""
    out_dir = str(tmp_path / "features")
    n1 = feature_engineer_from_config(synthetic_cfg, output_dir=out_dir)
    n2 = feature_engineer_from_config(synthetic_cfg, output_dir=out_dir)
    assert n1 == n2
