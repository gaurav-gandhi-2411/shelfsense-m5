"""Integration tests: materialize each model asset in test_mode.

Full upstream chain for multi-horizon variants (tvp_13/17, rmse_mh, ylags):
  raw_sales/calendar/prices -> raw_validated -> features(100 series) ->
  features_validated -> model_<variant>

For model_store_dept: features test_mode=False so feature_engineer skips
existing store parquets (completing in ~1s) and model_store_dept trains
2 slices with 10 boost rounds.

Skip when MLflow is not reachable. Run after: docker compose up -d

Each test should complete in < 90s.
"""
from __future__ import annotations

import os

import pytest
import requests


TRACKING_URI = "http://localhost:5000"
RAW_DIR      = "data/raw/m5-forecasting-accuracy"
FEATURES_DIR = "data/processed/features"


def _mlflow_reachable() -> bool:
    try:
        r = requests.get(f"{TRACKING_URI}/health", timeout=2)
        return r.status_code == 200 and r.text.strip() == "OK"
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _mlflow_reachable(),
    reason="MLflow not reachable at localhost:5000 — start with: docker compose up -d",
)


def _upstream_assets():
    from shelfsense.orchestration.assets import (
        features,
        features_validated,
        raw_calendar,
        raw_prices,
        raw_sales,
        raw_validated,
    )
    return [raw_sales, raw_calendar, raw_prices, raw_validated, features, features_validated]


def _test_features_run_config(test_mode: bool = True) -> dict:
    return {
        "raw_sales":    {"config": {"raw_dir": RAW_DIR}},
        "raw_calendar": {"config": {"raw_dir": RAW_DIR}},
        "raw_prices":   {"config": {"raw_dir": RAW_DIR}},
        "features": {
            "config": {
                "output_dir":    FEATURES_DIR,
                "last_day":      1941,
                "test_mode":     test_mode,
                "test_n_series": 100,
                "test_seed":     42,
            }
        },
    }


def _check_mlflow_run(asset_name: str) -> None:
    import mlflow
    mlflow.set_tracking_uri(TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name("shelfsense-m5")
    assert exp is not None, "shelfsense-m5 experiment not found"
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=f"tags.asset = '{asset_name}'",
        max_results=1,
    )
    assert len(runs) > 0, f"No MLflow run found for {asset_name}"
    assert "val_wrmsse" in runs[0].data.metrics, "val_wrmsse metric missing"
    assert runs[0].data.params.get("test_mode") == "True"


# ---------------------------------------------------------------------------
# model_tvp_13
# ---------------------------------------------------------------------------

def test_model_tvp_13_test_mode():
    """Materialize model_tvp_13 via the full upstream chain in test_mode."""
    from dagster import materialize
    from shelfsense.orchestration.assets import model_tvp_13
    from shelfsense.orchestration.resources import MLflowResource

    model_dir = "data/models/test_tvp_1p3"
    run_config = {
        "ops": {
            **_test_features_run_config(test_mode=True),
            "model_tvp_13": {
                "config": {"model_dir": model_dir, "raw_dir": RAW_DIR, "test_mode": True}
            },
        }
    }

    result = materialize(
        assets=_upstream_assets() + [model_tvp_13],
        resources={"mlflow_resource": MLflowResource(tracking_uri=TRACKING_URI)},
        run_config=run_config,
    )
    assert result.success

    output = result.output_for_node("model_tvp_13")
    assert "val_wrmsse" in output
    assert 0 < output["val_wrmsse"] < 10.0
    assert os.path.exists(os.path.join(model_dir, "h_01.pkl"))
    _check_mlflow_run("model_tvp_13")


# ---------------------------------------------------------------------------
# model_tvp_17
# ---------------------------------------------------------------------------

def test_model_tvp_17_test_mode():
    """Materialize model_tvp_17 via the full upstream chain in test_mode."""
    from dagster import materialize
    from shelfsense.orchestration.assets import model_tvp_17
    from shelfsense.orchestration.resources import MLflowResource

    model_dir = "data/models/test_tvp_1p7"
    run_config = {
        "ops": {
            **_test_features_run_config(test_mode=True),
            "model_tvp_17": {
                "config": {"model_dir": model_dir, "raw_dir": RAW_DIR, "test_mode": True}
            },
        }
    }

    result = materialize(
        assets=_upstream_assets() + [model_tvp_17],
        resources={"mlflow_resource": MLflowResource(tracking_uri=TRACKING_URI)},
        run_config=run_config,
    )
    assert result.success

    output = result.output_for_node("model_tvp_17")
    assert "val_wrmsse" in output
    assert 0 < output["val_wrmsse"] < 10.0
    assert os.path.exists(os.path.join(model_dir, "h_01.pkl"))
    _check_mlflow_run("model_tvp_17")


# ---------------------------------------------------------------------------
# model_rmse_mh
# ---------------------------------------------------------------------------

def test_model_rmse_mh_test_mode():
    """Materialize model_rmse_mh via the full upstream chain in test_mode."""
    from dagster import materialize
    from shelfsense.orchestration.assets import model_rmse_mh
    from shelfsense.orchestration.resources import MLflowResource

    model_dir = "data/models/test_rmse_mh"
    run_config = {
        "ops": {
            **_test_features_run_config(test_mode=True),
            "model_rmse_mh": {
                "config": {"model_dir": model_dir, "raw_dir": RAW_DIR, "test_mode": True}
            },
        }
    }

    result = materialize(
        assets=_upstream_assets() + [model_rmse_mh],
        resources={"mlflow_resource": MLflowResource(tracking_uri=TRACKING_URI)},
        run_config=run_config,
    )
    assert result.success

    output = result.output_for_node("model_rmse_mh")
    assert "val_wrmsse" in output
    assert 0 < output["val_wrmsse"] < 10.0
    assert os.path.exists(os.path.join(model_dir, "h_01.pkl"))
    _check_mlflow_run("model_rmse_mh")


# ---------------------------------------------------------------------------
# model_store_dept
# ---------------------------------------------------------------------------

def test_model_store_dept_test_mode():
    """Materialize model_store_dept in test_mode (1 slice, 100-series features_test).

    features run in test_mode=True (100 sampled series → features_test dir).
    model_store_dept trains CA_1×FOODS_1 only (20 series in features_test seed=42).
    """
    from dagster import materialize
    from shelfsense.orchestration.assets import model_store_dept
    from shelfsense.orchestration.resources import MLflowResource

    model_dir = "data/models/test_store_dept"
    run_config = {
        "ops": {
            **_test_features_run_config(test_mode=True),
            "model_store_dept": {
                "config": {"model_dir": model_dir, "raw_dir": RAW_DIR, "test_mode": True}
            },
        }
    }

    result = materialize(
        assets=_upstream_assets() + [model_store_dept],
        resources={"mlflow_resource": MLflowResource(tracking_uri=TRACKING_URI)},
        run_config=run_config,
    )
    assert result.success

    output = result.output_for_node("model_store_dept")
    assert "val_wrmsse" in output
    assert 0 < output["val_wrmsse"] < 10.0

    import glob as glob_mod
    pkls = glob_mod.glob(os.path.join(model_dir, "lgbm_SD_*.pkl"))
    assert len(pkls) >= 1, f"No pkl files found in {model_dir}"
    _check_mlflow_run("model_store_dept")


# ---------------------------------------------------------------------------
# model_ylags
# ---------------------------------------------------------------------------

def test_model_ylags_test_mode():
    """Materialize model_ylags via the full upstream chain in test_mode.

    Uses same features as model_tvp_13 — annual lags already present on disk.
    """
    from dagster import materialize
    from shelfsense.orchestration.assets import model_ylags
    from shelfsense.orchestration.resources import MLflowResource

    model_dir = "data/models/test_ylags"
    run_config = {
        "ops": {
            **_test_features_run_config(test_mode=True),
            "model_ylags": {
                "config": {"model_dir": model_dir, "raw_dir": RAW_DIR, "test_mode": True}
            },
        }
    }

    result = materialize(
        assets=_upstream_assets() + [model_ylags],
        resources={"mlflow_resource": MLflowResource(tracking_uri=TRACKING_URI)},
        run_config=run_config,
    )
    assert result.success

    output = result.output_for_node("model_ylags")
    assert "val_wrmsse" in output
    assert 0 < output["val_wrmsse"] < 10.0
    assert os.path.exists(os.path.join(model_dir, "h_01.pkl"))
    _check_mlflow_run("model_ylags")
