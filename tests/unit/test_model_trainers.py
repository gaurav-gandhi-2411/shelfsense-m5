"""Unit tests for LightGBM model trainer modules.

All tests run without training data or ML infrastructure — they cover
initialization, pure helper functions, and edge-case behavior on
synthetic DataFrames.
"""

from __future__ import annotations

import pandas as pd

# ── multihorizon.py constants ─────────────────────────────────────────────────


def test_mht_default_feature_cols_length():
    from shelfsense.models.lightgbm.multihorizon import DEFAULT_FEATURE_COLS

    assert len(DEFAULT_FEATURE_COLS) == 42  # 38 numeric + 4 categorical


def test_mht_ylags_feature_cols_includes_annual_lags():
    from shelfsense.models.lightgbm.multihorizon import YLAGS_FEATURE_COLS

    for lag in ("lag_91", "lag_182", "lag_364"):
        assert lag in YLAGS_FEATURE_COLS


# ── MultiHorizonTrainer.__init__ ──────────────────────────────────────────────


def test_mht_init_tweedie_params():
    from shelfsense.models.lightgbm.multihorizon import MultiHorizonTrainer

    cfg = {"objective": "tweedie", "tvp": 1.5, "num_boost_round": 100}
    t = MultiHorizonTrainer(cfg)
    assert t._lgb_params["objective"] == "tweedie"
    assert t._lgb_params["metric"] == "tweedie"
    assert t._lgb_params["tweedie_variance_power"] == 1.5
    assert t.num_boost_round == 100


def test_mht_init_regression_no_tvp():
    from shelfsense.models.lightgbm.multihorizon import MultiHorizonTrainer

    t = MultiHorizonTrainer({"objective": "regression"})
    assert t._lgb_params["metric"] == "rmse"
    assert "tweedie_variance_power" not in t._lgb_params


def test_mht_horizon_override_stored():
    from shelfsense.models.lightgbm.multihorizon import MultiHorizonTrainer

    t = MultiHorizonTrainer({"objective": "tweedie", "horizon": 5})
    assert t.horizon == 5


def test_mht_early_stopping_default():
    from shelfsense.models.lightgbm.multihorizon import MultiHorizonTrainer

    t = MultiHorizonTrainer({"objective": "tweedie"})
    assert t.early_stopping_rounds == 75


# ── store_dept.py helpers ─────────────────────────────────────────────────────


def test_sdt_design_hash_deterministic():
    from shelfsense.models.lightgbm.store_dept import _compute_design_hash

    cfg = {"objective": "tweedie", "tvp": 1.3, "optuna_trials": 10}
    assert _compute_design_hash(cfg) == _compute_design_hash(cfg)


def test_sdt_design_hash_changes_on_tvp():
    from shelfsense.models.lightgbm.store_dept import _compute_design_hash

    h13 = _compute_design_hash({"objective": "tweedie", "tvp": 1.3})
    h17 = _compute_design_hash({"objective": "tweedie", "tvp": 1.7})
    assert h13 != h17


def test_sdt_model_filename_format():
    from shelfsense.models.lightgbm.store_dept import _model_filename

    fname = _model_filename("CA_1", "FOODS_1", "abcd1234")
    assert fname == "lgbm_SD_CA_1_FOODS_1_pabcd1234.pkl"


def test_sdt_build_hist_from_wide_shape_and_range():
    from shelfsense.models.lightgbm.store_dept import _build_hist_from_wide

    n_days = 220
    sales = pd.DataFrame(
        [
            {
                "id": "FOODS_1_001_CA_1_evaluation",
                "item_id": "FOODS_1_001",
                "cat_id": "FOODS",
                "dept_id": "FOODS_1",
                "store_id": "CA_1",
                "state_id": "CA",
                **{f"d_{d}": d % 5 for d in range(1, n_days + 1)},
            }
        ]
    )
    result = _build_hist_from_wide(
        sales,
        series_ids=["FOODS_1_001_CA_1_evaluation"],
        last_day=220,
        history_days=50,
    )
    assert set(result.columns) >= {"id", "d_num", "sales"}
    assert len(result) == 50
    assert result["d_num"].min() == 171
    assert result["d_num"].max() == 220


def test_sdt_val_preds_from_cache_empty_on_no_pkls(tmp_path):
    from shelfsense.models.lightgbm.store_dept import StoreDeptTrainer

    trainer = StoreDeptTrainer(
        {
            "objective": "tweedie",
            "stores": ["CA_1"],
            "departments": ["FOODS_1"],
        }
    )
    result = trainer.val_preds_from_cache(str(tmp_path))
    assert list(result.columns) == ["id"] + [f"F{h}" for h in range(1, 29)]
    assert len(result) == 0


# ── recursive.py constants ────────────────────────────────────────────────────


def test_recursive_lags_and_windows():
    from shelfsense.models.lightgbm.recursive import LAGS, WINDOWS

    assert LAGS == [7, 14, 28, 56]
    assert WINDOWS == [7, 28, 56, 180]


def test_recursive_num_features_composition():
    from shelfsense.models.lightgbm.recursive import (
        CAL_FEATURE_COLS,
        LAG_FEATURES,
        NUM_FEATURES,
        PRICE_FEATURE_COLS,
        ROLL_FEATURES,
    )

    assert len(CAL_FEATURE_COLS) == 13
    assert len(PRICE_FEATURE_COLS) == 5
    assert len(LAG_FEATURES) == 4
    assert len(ROLL_FEATURES) == 16
    assert len(NUM_FEATURES) == 38


def test_recursive_all_features_count():
    from shelfsense.models.lightgbm.recursive import ALL_FEATURES, CAT_COLS, NUM_FEATURES

    assert len(CAT_COLS) == 4
    assert len(ALL_FEATURES) == len(NUM_FEATURES) + len(CAT_COLS)


def test_recursive_history_days_covers_max_window():
    from shelfsense.models.lightgbm.recursive import HISTORY_DAYS, WINDOWS

    assert HISTORY_DAYS >= max(WINDOWS)


def test_recursive_build_history_df_shape_and_range():
    """_build_history_df melts wide sales into long format with d_num."""
    from shelfsense.models.lightgbm.recursive import HISTORY_DAYS, _build_history_df
    from tests.fixtures.synthetic_m5 import make_sales_df

    n_days = HISTORY_DAYS + 50
    sales_df = make_sales_df(n_days=n_days, stores=["CA_1"], n_items_per_store=2)
    series_ids = sales_df["id"].values

    result = _build_history_df(sales_df, series_ids, last_day=n_days, history_days=HISTORY_DAYS)

    assert "d_num" in result.columns
    assert "sales" in result.columns
    assert "d" not in result.columns
    assert len(result) == HISTORY_DAYS * len(series_ids)
    assert result["d_num"].min() == n_days - HISTORY_DAYS + 1
    assert result["d_num"].max() == n_days


# ── store_dept.py val_preds_from_cache non-empty ──────────────────────────────


def test_recursive_build_price_by_day():
    """_build_price_by_day returns a dict of day → (n_series, 5) price arrays."""
    from shelfsense.features.calendar import build_calendar_lookup
    from shelfsense.features.price import build_price_lookup
    from shelfsense.models.lightgbm.recursive import PRICE_FEATURE_COLS, _build_price_by_day
    from tests.fixtures.synthetic_m5 import make_calendar_df, make_prices_df, make_sales_df

    n_days = 240
    sales = make_sales_df(n_days=n_days, stores=["CA_1"], n_items_per_store=2)
    cal_df = make_calendar_df(n_days=n_days)
    prices_df = make_prices_df(sales, cal_df)

    cal_lookup = build_calendar_lookup(cal_df)
    price_lookup = build_price_lookup(prices_df, cal_df)

    series_meta = sales[["item_id", "store_id"]].drop_duplicates().reset_index(drop=True)

    start_day, end_day = 211, 218
    result = _build_price_by_day(series_meta, price_lookup, cal_lookup, start_day, end_day)

    assert set(result.keys()) == set(range(start_day, end_day + 1))
    for arr in result.values():
        assert arr.shape == (len(series_meta), len(PRICE_FEATURE_COLS))


def test_mht_fit_accepts_store_and_dept_filter():
    """MultiHorizonTrainer.fit() exposes store_filter and dept_filter kwargs."""
    import inspect

    from shelfsense.models.lightgbm.multihorizon import MultiHorizonTrainer

    sig = inspect.signature(MultiHorizonTrainer.fit)
    assert "store_filter" in sig.parameters
    assert "dept_filter" in sig.parameters


def test_mht_predict_accepts_store_and_dept_filter():
    """MultiHorizonTrainer.predict() exposes store_filter and dept_filter kwargs."""
    import inspect

    from shelfsense.models.lightgbm.multihorizon import MultiHorizonTrainer

    sig = inspect.signature(MultiHorizonTrainer.predict)
    assert "store_filter" in sig.parameters
    assert "dept_filter" in sig.parameters


def test_parse_slices_all_returns_none():
    """_parse_slices('all') returns None (meaning 'train all slices')."""
    from shelfsense.orchestration.assets import _parse_slices

    assert _parse_slices("all") is None


def test_parse_slices_single_key():
    """_parse_slices parses a valid STORE_DEPT key into a list of tuples."""
    from shelfsense.orchestration.assets import _parse_slices

    result = _parse_slices("CA_1_FOODS_3")
    assert result == [("CA_1", "FOODS_3")]


def test_parse_slices_multiple_keys():
    """_parse_slices handles comma-separated keys."""
    from shelfsense.orchestration.assets import _parse_slices

    result = _parse_slices("CA_1_FOODS_3,TX_2_HOBBIES_1")
    assert result == [("CA_1", "FOODS_3"), ("TX_2", "HOBBIES_1")]


def test_parse_slices_invalid_key_skipped():
    """_parse_slices ignores keys that don't match a known store×dept pair."""
    from shelfsense.orchestration.assets import _parse_slices

    result = _parse_slices("ZZ_1_INVALID")
    assert result is None  # falls through to `return result or None`


def test_sdt_val_preds_from_cache_returns_rows_from_pkl(tmp_path):
    """val_preds_from_cache reads pkl and returns DataFrame with F1..F28."""
    import pickle

    import numpy as np

    from shelfsense.models.lightgbm.store_dept import HORIZON, StoreDeptTrainer, _model_filename

    trainer = StoreDeptTrainer(
        {
            "objective": "tweedie",
            "stores": ["CA_1"],
            "departments": ["FOODS_1"],
        }
    )
    series_ids = ["FOODS_1_001_CA_1_evaluation", "FOODS_1_002_CA_1_evaluation"]
    val_preds = np.ones((2, HORIZON), dtype=np.float32)

    pkl_path = tmp_path / _model_filename("CA_1", "FOODS_1", trainer.design_hash)
    with open(pkl_path, "wb") as fh:
        pickle.dump(
            {
                "series_ids": series_ids,
                "val_preds": val_preds,
                "store": "CA_1",
                "dept": "FOODS_1",
                "n_series": 2,
                "val_tweedie": 0.5,
                "val_wrmsse": 0.7,
                "best_iter": 100,
                "best_params": {},
            },
            fh,
        )

    result = trainer.val_preds_from_cache(str(tmp_path))
    assert len(result) == 2
    assert list(result.columns) == ["id"] + [f"F{h}" for h in range(1, HORIZON + 1)]
    assert (result["F1"] == 1.0).all()
