"""Integration smoke test for MLflowResource.

Skipped automatically when MLflow is not reachable at localhost:5000.
Run after: docker compose up -d
"""

from __future__ import annotations

import pytest
import requests

TRACKING_URI = "http://localhost:5000"


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


def test_mlflow_resource_smoke():
    import mlflow

    from shelfsense.orchestration.resources import MLflowResource

    resource = MLflowResource(tracking_uri=TRACKING_URI)

    exp_id = resource.get_experiment("shelfsense-m5-test")
    assert exp_id is not None

    run_id = resource.log_asset_run(
        run_name="integration_smoke",
        metrics={"smoke_metric": 42.0},
        params={"smoke_param": "hello"},
        tags={"test": "true"},
    )
    assert run_id is not None

    mlflow.set_tracking_uri(TRACKING_URI)
    run = mlflow.get_run(run_id)
    assert run.data.metrics["smoke_metric"] == 42.0
    assert run.data.params["smoke_param"] == "hello"
    assert run.data.tags["test"] == "true"


def test_get_experiment_idempotent():
    from shelfsense.orchestration.resources import MLflowResource

    resource = MLflowResource(tracking_uri=TRACKING_URI)
    exp_id_1 = resource.get_experiment("shelfsense-m5-test")
    exp_id_2 = resource.get_experiment("shelfsense-m5-test")
    assert exp_id_1 == exp_id_2


def test_log_metrics_to_run():
    import mlflow

    from shelfsense.orchestration.resources import MLflowResource

    resource = MLflowResource(tracking_uri=TRACKING_URI)
    mlflow.set_tracking_uri(TRACKING_URI)
    exp_id = resource.get_experiment("shelfsense-m5-test")

    with mlflow.start_run(experiment_id=exp_id, run_name="metrics_append_test") as run:
        run_id = run.info.run_id

    resource.log_metrics_to_run(run_id, {"appended_metric": 7.0})

    run_data = mlflow.get_run(run_id)
    assert run_data.data.metrics["appended_metric"] == 7.0
