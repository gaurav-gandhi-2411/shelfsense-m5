"""
Unit tests for the WRMSSE evaluator.
"""
import numpy as np
import pandas as pd
import pytest
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluation.wrmsse import (
    LEVEL_SPECS,
    HORIZON,
    build_scales,
    build_revenue_weights,
    compute_rmsse_per_series,
    compute_wrmsse,
    submission_to_matrix,
    _group_keys,
    _aggregate,
)


# ── synthetic fixture ─────────────────────────────────────────────────────────

def _make_tiny_sales(n_series: int = 10, n_train: int = 50, seed: int = 0):
    """Create a minimal synthetic sales dataset for testing."""
    rng = np.random.default_rng(seed)

    stores  = ["CA_1", "TX_1"]
    states  = {"CA_1": "CA", "TX_1": "TX"}
    cats    = ["FOODS", "HOBBIES"]
    depts   = {"FOODS": "FOODS_1", "HOBBIES": "HOBBIES_1"}
    items   = [f"ITEM_{i:03d}" for i in range(n_series)]

    rows = []
    for i, item in enumerate(items):
        store = stores[i % 2]
        cat   = cats[i % 2]
        dept  = depts[cat]
        state = states[store]
        sid   = f"{item}_{store}_validation"
        sales = rng.integers(0, 5, size=n_train).tolist()
        rows.append([sid, item, dept, cat, store, state] + sales)

    day_cols  = [f"d_{d}" for d in range(1, n_train + 1)]
    meta_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    df = pd.DataFrame(rows, columns=meta_cols + day_cols)
    df[day_cols] = df[day_cols].astype(np.int16)

    # Calendar: one row per day
    cal = pd.DataFrame({
        "d":       day_cols,
        "wm_yr_wk": [1101 + d // 7 for d in range(n_train)],
        "date":    pd.date_range("2011-01-29", periods=n_train),
    })

    # Prices: one entry per (store, item, week)
    weeks = cal["wm_yr_wk"].unique()
    price_rows = []
    for store in stores:
        for item in items:
            for wk in weeks:
                price_rows.append([store, item, wk, rng.uniform(1, 10)])
    prices = pd.DataFrame(price_rows, columns=["store_id", "item_id", "wm_yr_wk", "sell_price"])

    return df, cal, prices, n_train


# ── build_scales ──────────────────────────────────────────────────────────────

def test_build_scales_random_walk():
    rng = np.random.default_rng(42)
    mat = np.cumsum(rng.standard_normal((5, 200)), axis=1)
    scales = build_scales(mat)
    assert scales.shape == (5,)
    assert np.all(scales > 0)


def test_build_scales_constant_nonzero_series():
    # Constant non-zero series: first non-zero found, diffs all 0 → scale=0
    mat = np.ones((3, 100))
    scales = build_scales(mat)
    assert np.allclose(scales, 0.0)


def test_build_scales_all_zero_series():
    mat = np.zeros((3, 100))
    scales = build_scales(mat)
    # All-zero → scale=0 (RMSSE will be inf, filtered downstream)
    assert np.allclose(scales, 0.0)


def test_build_scales_leading_zeros_trimmed():
    # First 50 days are zero, last 50 days are a ramp y_t=t → diffs=1 → scale=1.0
    mat = np.zeros((1, 100))
    mat[0, 50:] = np.arange(50, dtype=float)
    scales = build_scales(mat)
    assert np.isclose(scales[0], 1.0), f"Expected 1.0, got {scales[0]}"


def test_build_scales_known_value():
    # y_t = t, so diff = 1 everywhere, mean(diff^2) = 1.0
    mat = np.arange(100).reshape(1, 100).astype(float)
    scales = build_scales(mat)
    assert np.isclose(scales[0], 1.0)


# ── build_revenue_weights ─────────────────────────────────────────────────────

def test_weights_sum_to_one():
    df, cal, prices, n_train = _make_tiny_sales(n_series=10, n_train=50)
    w_dict = build_revenue_weights(df, prices, cal, last_train_day=n_train)

    for level, (keys, w) in w_dict.items():
        assert np.isclose(w.sum(), 1.0, atol=1e-9), (
            f"Level {level}: weights sum to {w.sum():.6f}, not 1.0"
        )


def test_weights_nonnegative():
    df, cal, prices, n_train = _make_tiny_sales(n_series=10, n_train=50)
    w_dict = build_revenue_weights(df, prices, cal, last_train_day=n_train)
    for level, (keys, w) in w_dict.items():
        assert np.all(w >= 0), f"Level {level} has negative weights"


# ── compute_wrmsse ────────────────────────────────────────────────────────────

def test_perfect_forecast_gives_zero_wrmsse():
    df, cal, prices, n_train = _make_tiny_sales(n_series=10, n_train=100)
    actuals = df[[f"d_{d}" for d in range(n_train - HORIZON + 1, n_train + 1)]].values.astype(float)
    preds   = actuals.copy()
    train_df = df.copy()

    wrmsse, level_scores = compute_wrmsse(
        preds, actuals, train_df, prices, cal, last_train_day=n_train - HORIZON
    )
    assert np.isclose(wrmsse, 0.0, atol=1e-9), f"Expected 0, got {wrmsse}"
    for lv, score in level_scores.items():
        assert np.isclose(score, 0.0, atol=1e-9), f"Level {lv}: expected 0, got {score}"


def test_wrmsse_positive_for_nonzero_error():
    df, cal, prices, n_train = _make_tiny_sales(n_series=10, n_train=100)
    actuals = df[[f"d_{d}" for d in range(n_train - HORIZON + 1, n_train + 1)]].values.astype(float)
    preds   = actuals + 1.0    # constant bias

    wrmsse, _ = compute_wrmsse(
        preds, actuals, df, prices, cal, last_train_day=n_train - HORIZON
    )
    assert wrmsse > 0.0


def test_twelve_level_breakdown():
    df, cal, prices, n_train = _make_tiny_sales(n_series=10, n_train=100)
    actuals = df[[f"d_{d}" for d in range(n_train - HORIZON + 1, n_train + 1)]].values.astype(float)
    preds   = np.zeros_like(actuals)

    wrmsse, level_scores = compute_wrmsse(
        preds, actuals, df, prices, cal, last_train_day=n_train - HORIZON
    )
    assert len(level_scores) == 12
    assert set(level_scores.keys()) == set(LEVEL_SPECS.keys())
    # total must equal unweighted mean of level scores
    assert np.isclose(wrmsse, np.mean(list(level_scores.values())), atol=1e-9)


def test_all_zero_actuals_and_preds():
    df, cal, prices, n_train = _make_tiny_sales(n_series=6, n_train=60)
    actuals = np.zeros((6, HORIZON))
    preds   = np.zeros((6, HORIZON))
    wrmsse, _ = compute_wrmsse(
        preds, actuals, df, prices, cal, last_train_day=n_train - HORIZON
    )
    assert np.isclose(wrmsse, 0.0, atol=1e-9)


# ── compute_rmsse_per_series ──────────────────────────────────────────────────

def test_rmsse_per_series_shape():
    n = 30
    preds   = np.random.rand(n, HORIZON)
    actuals = np.random.rand(n, HORIZON)
    scales  = np.ones(n)
    rmsse = compute_rmsse_per_series(preds, actuals, scales)
    assert rmsse.shape == (n,)
    assert np.all(rmsse >= 0)


def test_rmsse_naive_on_random_walk_expected_range():
    """
    For a random walk with iid N(0,1) steps, the naive (last-value) forecast
    accumulates error over the horizon:
      E[MSE] = mean(h for h=1..28) = 14.5
      E[scale] = 1.0
      E[RMSSE] = sqrt(14.5) ≈ 3.81

    We average over 200 series to reduce variance and test for that range.
    """
    rng = np.random.default_rng(7)
    n_series = 200
    noise = rng.standard_normal((n_series, 1000))
    series = np.cumsum(noise, axis=1)
    train   = series[:, :-HORIZON]
    actuals = series[:, -HORIZON:]
    preds   = np.tile(train[:, -1:], (1, HORIZON))

    scales = build_scales(train)
    rmsse  = compute_rmsse_per_series(preds, actuals, scales)
    mean_rmsse = rmsse.mean()

    # Expected ≈ 3.81 (sqrt(14.5)); allow generous tolerance for finite samples
    assert 2.5 < mean_rmsse < 5.5, f"Mean RMSSE={mean_rmsse:.3f} outside expected range"


# ── submission_to_matrix ──────────────────────────────────────────────────────

def test_submission_to_matrix_alignment():
    df, _, _, _ = _make_tiny_sales(n_series=4, n_train=30)
    fcols = [f"F{i}" for i in range(1, HORIZON + 1)]
    rng   = np.random.default_rng(0)
    vals  = rng.integers(0, 10, size=(4, HORIZON)).astype(float)

    sub_rows = pd.DataFrame(
        np.column_stack([df["id"].values, vals]),
        columns=["id"] + fcols,
    )
    sub_rows[fcols] = sub_rows[fcols].astype(float)

    mat = submission_to_matrix(sub_rows, df, suffix="_validation")
    assert mat.shape == (4, HORIZON)
    assert np.allclose(mat, vals, atol=1e-6)
