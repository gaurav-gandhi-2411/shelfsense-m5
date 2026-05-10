"""Integration tests for M5Dataset schema validation and the data validate CLI command.

Non-gated tests use the synthetic fixture and run without M5 data.
Real-data tests are gated by RUN_REAL_DATA_TESTS=1; they also require:
- M5 CSVs at data/raw/m5-forecasting-accuracy/
- Feature parquets at data/processed/features/
- MLflow reachable (soft — assets skip log on failure)
"""
from __future__ import annotations

import os

import pytest

RAW_DIR      = "data/raw/m5-forecasting-accuracy"
FEATURES_DIR = "data/processed/features"

_real_data = pytest.mark.skipif(
    os.environ.get("RUN_REAL_DATA_TESTS") != "1",
    reason=(
        "Real-data tests disabled — set RUN_REAL_DATA_TESTS=1. "
        "Requires M5 CSVs + feature parquets."
    ),
)


# ── synthetic (no env var required) ──────────────────────────────────────────

def test_m5dataset_raw_validate_synthetic(tmp_path):
    """Pandera raw schemas pass on synthetic M5-shaped CSVs."""
    from tests.fixtures.synthetic_m5 import write_synthetic_csvs

    from shelfsense.data.load import M5Dataset

    raw_dir = write_synthetic_csvs(str(tmp_path / "raw"))
    ds = M5Dataset(raw_dir=raw_dir, features_dir="", validate=False)
    results = ds.validate_raw()
    failures = {k: v for k, v in results.items() if not k.endswith("__error") and not v}
    assert not failures, f"Synthetic raw validation failures: {failures}"


def test_m5dataset_prices_property_synthetic(tmp_path):
    """M5Dataset.prices returns expected columns on synthetic data."""
    from tests.fixtures.synthetic_m5 import write_synthetic_csvs

    from shelfsense.data.load import M5Dataset

    raw_dir = write_synthetic_csvs(str(tmp_path / "raw"))
    ds = M5Dataset(raw_dir=raw_dir, features_dir="", validate=False)
    prices = ds.prices
    assert set(prices.columns) == {"store_id", "item_id", "wm_yr_wk", "sell_price"}
    assert (prices["sell_price"] > 0).all()


# ── real-data (gated) ─────────────────────────────────────────────────────────

@_real_data
def test_m5dataset_raw_validate():
    """Pandera schemas pass on the real M5 CSVs."""
    from shelfsense.data.load import M5Dataset

    ds = M5Dataset(raw_dir=RAW_DIR, features_dir=FEATURES_DIR, validate=True)
    results = ds.validate_raw()
    failures = {k: v for k, v in results.items() if not k.endswith("__error") and not v}
    assert not failures, f"Raw validation failures: {failures}"


@_real_data
def test_m5dataset_feature_validate_single_store():
    """Validate one store parquet to avoid loading all 905 MB."""
    import glob

    from shelfsense.data.load import M5Dataset

    paths = sorted(glob.glob(os.path.join(FEATURES_DIR, "*.parquet")))
    assert paths, "No feature parquets found"

    ds = M5Dataset(raw_dir=RAW_DIR, features_dir=FEATURES_DIR, validate=True)
    store = os.path.splitext(os.path.basename(paths[0]))[0]
    df = ds.load_features(store=store)
    assert df.shape[0] > 0
    assert "d_num" in df.columns


@_real_data
def test_data_validate_cli_exit_zero():
    """shelfsense data validate exits 0 when all real-data checks pass."""
    from typer.testing import CliRunner

    from shelfsense.cli import app

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["data", "validate", "--raw-dir", RAW_DIR, "--features-dir", FEATURES_DIR],
    )
    assert result.exit_code == 0, result.output
    assert "Validation passed" in result.output
