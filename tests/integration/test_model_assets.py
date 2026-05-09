"""Integration test: materialize model_tvp_13 in test_mode.

Runs the full upstream chain:
  raw_sales/calendar/prices -> raw_validated -> features -> features_validated -> model_tvp_13

Skipped automatically when MLflow is not reachable at localhost:5000.
Run after: docker compose up -d

Use a reduced series count (100) and test_mode on both features and model assets
so the full run completes in < 5 minutes.
"""
from __future__ import annotations

import os

import pytest
import requests


TRACKING_URI = "http://localhost:5000"
RAW_DIR      = "data/raw/m5-forecasting-accuracy"


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


def test_model_tvp_13_test_mode():
    """Materialize model_tvp_13 via the full upstream chain in test_mode."""
    import mlflow
    from dagster import materialize

    from shelfsense.orchestration.assets import (
        features,
        features_validated,
        model_tvp_13,
        raw_calendar,
        raw_prices,
        raw_sales,
        raw_validated,
    )
    from shelfsense.orchestration.resources import MLflowResource

    model_dir = "data/models/test_tvp_1p3"

    run_config = {
        "ops": {
            "raw_sales":    {"config": {"raw_dir": RAW_DIR}},
            "raw_calendar": {"config": {"raw_dir": RAW_DIR}},
            "raw_prices":   {"config": {"raw_dir": RAW_DIR}},
            "features": {
                "config": {
                    "output_dir":    "data/processed/features",
                    "last_day":      1941,
                    "test_mode":     True,
                    "test_n_series": 100,
                    "test_seed":     42,
                }
            },
            "model_tvp_13": {
                "config": {
                    "model_dir": model_dir,
                    "raw_dir":   RAW_DIR,
                    "test_mode": True,
                }
            },
        }
    }

    mlflow_res = MLflowResource(tracking_uri=TRACKING_URI)

    result = materialize(
        assets=[
            raw_sales, raw_calendar, raw_prices,
            raw_validated, features, features_validated,
            model_tvp_13,
        ],
        resources={"mlflow_resource": mlflow_res},
        run_config=run_config,
    )

    assert result.success, "Dagster materialize failed"

    # Check return value
    output = result.output_for_node("model_tvp_13")
    assert "model_dir"  in output
    assert "val_wrmsse" in output
    assert 0 < output["val_wrmsse"] < 10.0, f"val_wrmsse out of range: {output['val_wrmsse']}"

    # Check model file was written (at least h_01.pkl)
    assert os.path.exists(os.path.join(model_dir, "h_01.pkl")), "h_01.pkl not found"

    # Check MLflow run was logged
    mlflow.set_tracking_uri(TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name("shelfsense-m5")
    assert exp is not None, "shelfsense-m5 experiment not found"
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="tags.asset = 'model_tvp_13'",
        max_results=1,
    )
    assert len(runs) > 0, "No MLflow run found for model_tvp_13"
    run_data = runs[0].data
    assert "val_wrmsse" in run_data.metrics, "val_wrmsse metric missing from MLflow run"
    assert run_data.params.get("test_mode") == "True"
