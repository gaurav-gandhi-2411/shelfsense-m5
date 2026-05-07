"""
Convert ShelfSense features parquet to darts-compatible format.

Column taxonomy:
  target:           sales (y)
  past_covariates:  lag_*, roll_* (not knowable at forecast horizon)
  future_covariates: calendar + price columns (available for test period)
  id_cols:          id, item_id, dept_id, cat_id, store_id, state_id, d (dropped)
"""
from __future__ import annotations

import pandas as pd
from pathlib import Path

M5_START_DATE = pd.Timestamp("2011-01-29")  # d_1 in M5 calendar

PAST_COV_COLS = [
    "lag_7", "lag_14", "lag_28", "lag_56",
    "roll_mean_7", "roll_std_7", "roll_min_7", "roll_max_7",
    "roll_mean_28", "roll_std_28", "roll_min_28", "roll_max_28",
    "roll_mean_56", "roll_std_56", "roll_min_56", "roll_max_56",
    "roll_mean_180", "roll_std_180", "roll_min_180", "roll_max_180",
]

FUTURE_COV_COLS = [
    "weekday", "month", "quarter", "year", "day_of_month", "week_of_year",
    "is_weekend", "is_holiday", "is_snap_ca", "is_snap_tx", "is_snap_wi",
    "days_since_event", "days_until_next_event",
    "sell_price", "price_change_pct", "price_relative_mean",
    "price_volatility", "has_price_change",
]

ID_COLS = ["item_id", "dept_id", "cat_id", "store_id", "state_id", "d"]


def to_long_format(
    features_parquet_path: str,
    output_path: str | None = None,
    overwrite: bool = False,
    min_d_num: int = 181,
) -> pd.DataFrame:
    """Convert features parquet to long format with unique_id / ds / y columns.

    min_d_num=181: clip rows where rolling/lag features are still NaN.
    roll_mean_180 is the slowest-initializing feature; d_num >= 181 guarantees
    all 20 past-covariate columns are fully populated. fillna(0) is NOT used —
    it would teach the model "no lag = zero demand" which corrupts signal.

    Set min_d_num=1 to get the full unclipped series (e.g. for inference context).

    Caches to output_path (snappy parquet) to avoid recomputation.
    """
    if output_path and not overwrite and Path(output_path).exists():
        return pd.read_parquet(output_path)

    df = pd.read_parquet(features_parquet_path)

    if min_d_num > 1:
        df = df[df["d_num"] >= min_d_num].copy()

    # Price columns are NaN when an item has no listed price for a given week
    # (item off-shelf, not yet launched, etc.). 0 of 3049 series are ALL-NaN;
    # ~50% have partial NaN. Forward-fill then back-fill per series is standard
    # M5 practice — prices are sticky and the item does return at the same price.
    _price_cols = [
        "sell_price", "price_change_pct", "price_relative_mean",
        "price_volatility", "has_price_change",
    ]
    df[_price_cols] = df.groupby("id")[_price_cols].transform(
        lambda x: x.ffill().bfill()
    )

    df["ds"] = M5_START_DATE + pd.to_timedelta(df["d_num"] - 1, unit="D")
    df = df.rename(columns={"id": "unique_id", "sales": "y"})
    df = df.drop(columns=ID_COLS + ["d_num"])

    front = ["unique_id", "ds", "y"]
    rest = [c for c in df.columns if c not in front]
    df = df[front + rest]

    feature_cols = df.columns[3:]
    assert not df[feature_cols].isna().any().any(), (
        f"NaN in features after clip (min_d_num={min_d_num}). "
        "Check that all lag/rolling windows are fully populated at this cutoff."
    )

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, compression="snappy", index=False)

    return df


def to_darts_datasets(
    df: pd.DataFrame,
) -> tuple:
    """Convert long-format df to (target_list, past_cov_list, future_cov_list).

    Each list has one darts TimeSeries per unique series.
    Caller must have darts installed; import is deferred so the module can be
    imported without darts (e.g. for format-only use cases).
    """
    from darts import TimeSeries

    targets, past_covs, future_covs = [], [], []

    for uid, grp in df.groupby("unique_id", sort=False):
        grp = grp.sort_values("ds").set_index("ds")

        targets.append(
            TimeSeries.from_series(grp["y"], freq="D")
        )
        past_covs.append(
            TimeSeries.from_dataframe(grp[PAST_COV_COLS].astype("float32"), freq="D")
        )
        future_covs.append(
            TimeSeries.from_dataframe(grp[FUTURE_COV_COLS].astype("float32"), freq="D")
        )

    return targets, past_covs, future_covs
