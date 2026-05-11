"""Negative-path unit tests for Pandera schemas.

Each test constructs a minimal valid DataFrame, then breaks one invariant
and asserts that SchemaError is raised.
"""

import numpy as np
import pandas as pd
import pandera.pandas as pa
import pytest

from shelfsense.data.schemas import (
    feature_schema,
    predictions_schema,
    raw_calendar_schema,
    raw_prices_schema,
    raw_sales_schema,
    submission_schema,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_raw_sales(n: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": ["FOODS_1_001_CA_1_evaluation"] * n,
            "item_id": ["FOODS_1_001"] * n,
            "dept_id": ["FOODS_1"] * n,
            "cat_id": ["FOODS"] * n,
            "store_id": ["CA_1"] * n,
            "state_id": ["CA"] * n,
            "d_1": [0.0] * n,
            "d_2": [1.0] * n,
        }
    )


def _make_raw_calendar(n: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2011-01-29"] * n,
            "wm_yr_wk": [11101] * n,
            "weekday": ["Saturday"] * n,
            "wday": [1] * n,
            "month": [1] * n,
            "year": [2011] * n,
            "d": [f"d_{i + 1}" for i in range(n)],
            "snap_CA": [0] * n,
            "snap_TX": [0] * n,
            "snap_WI": [0] * n,
        }
    )


def _make_raw_prices(n: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "store_id": ["CA_1"] * n,
            "item_id": ["FOODS_1_001"] * n,
            "wm_yr_wk": [11101] * n,
            "sell_price": [1.5] * n,
        }
    )


def _make_feature(n: int = 5) -> pd.DataFrame:
    lag_cols = {
        f"lag_{k}": np.full(n, np.nan, dtype="float32") for k in (7, 14, 28, 56, 91, 182, 364)
    }
    roll_cols = {
        f"roll_{s}_{w}": np.full(n, 1.0, dtype="float32")
        for s in ("mean", "std", "min", "max")
        for w in (7, 28, 56, 180)
    }
    return pd.DataFrame(
        {
            "id": ["FOODS_1_001_CA_1_evaluation"] * n,
            "item_id": ["FOODS_1_001"] * n,
            "dept_id": pd.Categorical(["FOODS_1"] * n),
            "cat_id": pd.Categorical(["FOODS"] * n),
            "store_id": pd.Categorical(["CA_1"] * n),
            "state_id": pd.Categorical(["CA"] * n),
            "d": [f"d_{i + 1}" for i in range(n)],
            "sales": np.ones(n, dtype="float32"),
            "d_num": np.arange(1, n + 1, dtype="int16"),
            "weekday": np.zeros(n, dtype="int8"),
            "month": np.ones(n, dtype="int8"),
            "quarter": np.ones(n, dtype="int8"),
            "year": np.full(n, 2011, dtype="int16"),
            "day_of_month": np.ones(n, dtype="int8"),
            "week_of_year": np.ones(n, dtype="int8"),
            "is_weekend": np.zeros(n, dtype="int8"),
            "is_holiday": np.zeros(n, dtype="int8"),
            "is_snap_ca": np.zeros(n, dtype="int8"),
            "is_snap_tx": np.zeros(n, dtype="int8"),
            "is_snap_wi": np.zeros(n, dtype="int8"),
            "days_since_event": np.full(n, np.nan, dtype="float32"),
            "days_until_next_event": np.full(n, np.nan, dtype="float32"),
            "sell_price": np.full(n, 1.5, dtype="float32"),
            "price_change_pct": np.zeros(n, dtype="float32"),
            "price_relative_mean": np.ones(n, dtype="float32"),
            "price_volatility": np.zeros(n, dtype="float32"),
            "has_price_change": np.zeros(n, dtype="int8"),
            **lag_cols,
            **roll_cols,
        }
    )


# ---------------------------------------------------------------------------
# Negative tests
# ---------------------------------------------------------------------------


def test_raw_sales_missing_required_column():
    df = _make_raw_sales().drop(columns=["store_id"])
    with pytest.raises(pa.errors.SchemaError):
        raw_sales_schema.validate(df)


def test_raw_calendar_invalid_month():
    df = _make_raw_calendar()
    df["month"] = 13
    with pytest.raises(pa.errors.SchemaError):
        raw_calendar_schema.validate(df)


def test_raw_calendar_invalid_snap_value():
    df = _make_raw_calendar()
    df["snap_CA"] = 2
    with pytest.raises(pa.errors.SchemaError):
        raw_calendar_schema.validate(df)


def test_raw_prices_negative_price():
    df = _make_raw_prices()
    df["sell_price"] = -0.5
    with pytest.raises(pa.errors.SchemaError):
        raw_prices_schema.validate(df)


def test_raw_prices_extra_column_rejected():
    df = _make_raw_prices()
    df["extra_col"] = 0
    with pytest.raises((pa.errors.SchemaError, pa.errors.SchemaErrors)):
        raw_prices_schema.validate(df)


def test_feature_schema_negative_d_num():
    df = _make_feature()
    df["d_num"] = df["d_num"] * -1
    with pytest.raises(pa.errors.SchemaError):
        feature_schema.validate(df)


def test_feature_schema_missing_column():
    df = _make_feature().drop(columns=["d_num"])
    with pytest.raises((pa.errors.SchemaError, pa.errors.SchemaErrors)):
        feature_schema.validate(df)


def test_submission_schema_negative_forecast():
    horizon = 28
    df = pd.DataFrame(
        {
            "id": ["FOODS_1_001_CA_1_evaluation"],
            **{f"F{h + 1}": [-0.1] for h in range(horizon)},
        }
    )
    with pytest.raises(pa.errors.SchemaError):
        submission_schema.validate(df)


def test_predictions_schema_missing_id():
    horizon = 28
    df = pd.DataFrame(
        {
            **{f"d_{1914 + h}": [1.0] for h in range(horizon)},
        }
    )
    with pytest.raises(pa.errors.SchemaError):
        predictions_schema.validate(df)
