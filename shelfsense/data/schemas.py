"""Pandera schemas for all ShelfSense data boundaries.

Public API
----------
raw_sales_schema      — sales_train_evaluation.csv
raw_calendar_schema   — calendar.csv
raw_prices_schema     — sell_prices.csv
feature_schema        — data/processed/features/<store>.parquet
predictions_schema    — model output parquet (id + horizon columns)
submission_schema     — Kaggle submission CSV (id + F1-F28)
"""

import pandera.pandas as pa
from pandera.pandas import Column, DataFrameSchema

# ---------------------------------------------------------------------------
# Raw input CSVs
# ---------------------------------------------------------------------------

raw_sales_schema = DataFrameSchema(
    columns={
        "id": Column(str, nullable=False),
        "item_id": Column(str, nullable=False),
        "dept_id": Column(str, nullable=False),
        "cat_id": Column(str, nullable=False),
        "store_id": Column(str, nullable=False),
        "state_id": Column(str, nullable=False),
        r"^d_\d+$": Column(pa.Float, nullable=True, regex=True, required=False),
    },
    name="raw_sales",
    strict=False,
    coerce=True,
)

raw_calendar_schema = DataFrameSchema(
    columns={
        "date": Column(str, nullable=False),
        "wm_yr_wk": Column(pa.Int, nullable=False, checks=pa.Check.gt(0)),
        "weekday": Column(str, nullable=False),
        "wday": Column(pa.Int, nullable=False, checks=pa.Check.in_range(1, 7)),
        "month": Column(pa.Int, nullable=False, checks=pa.Check.in_range(1, 12)),
        "year": Column(pa.Int, nullable=False, checks=pa.Check.gt(2000)),
        "d": Column(str, nullable=False),
        "event_name_1": Column(str, nullable=True, required=False),
        "event_type_1": Column(str, nullable=True, required=False),
        "event_name_2": Column(str, nullable=True, required=False),
        "event_type_2": Column(str, nullable=True, required=False),
        "snap_CA": Column(pa.Int, nullable=False, checks=pa.Check.isin([0, 1])),
        "snap_TX": Column(pa.Int, nullable=False, checks=pa.Check.isin([0, 1])),
        "snap_WI": Column(pa.Int, nullable=False, checks=pa.Check.isin([0, 1])),
    },
    name="raw_calendar",
    strict=False,
    coerce=True,
)

raw_prices_schema = DataFrameSchema(
    columns={
        "store_id": Column(str, nullable=False),
        "item_id": Column(str, nullable=False),
        "wm_yr_wk": Column(pa.Int, nullable=False, checks=pa.Check.gt(0)),
        "sell_price": Column(pa.Float, nullable=False, checks=pa.Check.gt(0)),
    },
    name="raw_prices",
    strict=True,
    coerce=True,
)

# ---------------------------------------------------------------------------
# Processed feature parquet
# ---------------------------------------------------------------------------

_LAG_COLS = {
    f"lag_{n}": Column(pa.Float, nullable=True, required=False)
    for n in (7, 14, 28, 56, 91, 182, 364)
}
_ROLL_COLS = {
    f"roll_{stat}_{w}": Column(pa.Float, nullable=True, required=False)
    for stat in ("mean", "std", "min", "max")
    for w in (7, 28, 56, 180)
}

feature_schema = DataFrameSchema(
    columns={
        "id": Column(str, nullable=False),
        "item_id": Column(str, nullable=False),
        "dept_id": Column(pa.Category, nullable=False),
        "cat_id": Column(pa.Category, nullable=False),
        "store_id": Column(pa.Category, nullable=False),
        "state_id": Column(pa.Category, nullable=False),
        "d": Column(str, nullable=False),
        "sales": Column(pa.Float, nullable=True),
        "d_num": Column(pa.Int, nullable=False, checks=pa.Check.ge(1)),
        "weekday": Column(pa.Int, nullable=False, checks=pa.Check.in_range(0, 6)),
        "month": Column(pa.Int, nullable=False, checks=pa.Check.in_range(1, 12)),
        "quarter": Column(pa.Int, nullable=False, checks=pa.Check.in_range(1, 4)),
        "year": Column(pa.Int, nullable=False, checks=pa.Check.gt(2000)),
        "day_of_month": Column(pa.Int, nullable=False, checks=pa.Check.in_range(1, 31)),
        "week_of_year": Column(pa.Int, nullable=False, checks=pa.Check.in_range(1, 53)),
        "is_weekend": Column(pa.Int, nullable=False, checks=pa.Check.isin([0, 1])),
        "is_holiday": Column(pa.Int, nullable=False, checks=pa.Check.isin([0, 1])),
        "is_snap_ca": Column(pa.Int, nullable=False, checks=pa.Check.isin([0, 1])),
        "is_snap_tx": Column(pa.Int, nullable=False, checks=pa.Check.isin([0, 1])),
        "is_snap_wi": Column(pa.Int, nullable=False, checks=pa.Check.isin([0, 1])),
        "days_since_event": Column(pa.Float, nullable=True),
        "days_until_next_event": Column(pa.Float, nullable=True),
        "sell_price": Column(pa.Float, nullable=True, checks=pa.Check.ge(0)),
        "price_change_pct": Column(pa.Float, nullable=True),
        "price_relative_mean": Column(pa.Float, nullable=True),
        "price_volatility": Column(pa.Float, nullable=True),
        "has_price_change": Column(pa.Int, nullable=False, checks=pa.Check.isin([0, 1])),
        **_LAG_COLS,
        **_ROLL_COLS,
    },
    name="features",
    strict=True,
    coerce=True,
)

# ---------------------------------------------------------------------------
# Model output and Kaggle submission
# ---------------------------------------------------------------------------

_HORIZON = 28
_PRED_COLS = {
    f"d_{1914 + h}": Column(pa.Float, nullable=False, checks=pa.Check.ge(0))
    for h in range(_HORIZON)
}
_SUB_COLS = {
    f"F{h + 1}": Column(pa.Float, nullable=False, checks=pa.Check.ge(0))
    for h in range(_HORIZON)
}

predictions_schema = DataFrameSchema(
    columns={
        "id": Column(str, nullable=False),
        **_PRED_COLS,
    },
    name="predictions",
    strict=True,
    coerce=True,
)

submission_schema = DataFrameSchema(
    columns={
        "id": Column(str, nullable=False),
        **_SUB_COLS,
    },
    name="submission",
    strict=True,
    coerce=True,
)
