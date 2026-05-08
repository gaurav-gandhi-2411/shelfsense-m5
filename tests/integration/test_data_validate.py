"""Integration tests for M5Dataset and shelfsense data validate.

These tests require the actual M5 data files to be present and DVC-pulled.
They are skipped automatically in CI or on machines without the data.
"""

import os

import pytest

RAW_DIR = "data/raw/m5-forecasting-accuracy"
FEATURES_DIR = "data/processed/features"

pytestmark = pytest.mark.skipif(
    not os.path.isdir(RAW_DIR) or not os.path.isdir(FEATURES_DIR),
    reason="M5 data not present — run `dvc pull` first",
)


def test_m5dataset_raw_validate():
    from shelfsense.data.load import M5Dataset

    ds = M5Dataset(raw_dir=RAW_DIR, features_dir=FEATURES_DIR, validate=True)
    results = ds.validate_raw()
    failures = {k: v for k, v in results.items() if not k.endswith("__error") and not v}
    assert not failures, f"Raw validation failures: {failures}"


def test_m5dataset_feature_validate_single_store():
    """Validate one store parquet to avoid loading all 905 MB in CI-like runs."""
    import glob

    from shelfsense.data.load import M5Dataset

    paths = sorted(glob.glob(os.path.join(FEATURES_DIR, "*.parquet")))
    assert paths, "No feature parquets found"

    ds = M5Dataset(raw_dir=RAW_DIR, features_dir=FEATURES_DIR, validate=True)
    store = os.path.splitext(os.path.basename(paths[0]))[0]
    df = ds.load_features(store=store)
    assert df.shape[0] > 0
    assert "d_num" in df.columns


def test_m5dataset_prices_property():
    from shelfsense.data.load import M5Dataset

    ds = M5Dataset(raw_dir=RAW_DIR, features_dir=FEATURES_DIR, validate=True)
    prices = ds.prices
    assert set(prices.columns) == {"store_id", "item_id", "wm_yr_wk", "sell_price"}
    assert (prices["sell_price"] > 0).all()


def test_data_validate_cli_exit_zero(tmp_path):
    """shelfsense data validate exits 0 when all files pass."""
    from typer.testing import CliRunner

    from shelfsense.cli import app

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["data", "validate", "--raw-dir", RAW_DIR, "--features-dir", FEATURES_DIR],
    )
    assert result.exit_code == 0, result.output
    assert "checks passed" in result.output
