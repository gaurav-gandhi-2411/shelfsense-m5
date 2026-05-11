"""Unit tests for MLflow tracking utilities and the Dagster MLflowResource.

All tests mock the MLflow client so no MLflow server is required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

# ── mlflow_utils.get_tracking_uri ─────────────────────────────────────────────


def test_get_tracking_uri_default(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    from shelfsense.tracking.mlflow_utils import get_tracking_uri

    uri = get_tracking_uri()
    assert "localhost" in uri and "5000" in uri


def test_get_tracking_uri_from_env(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://my-mlflow:9999")
    from shelfsense.tracking.mlflow_utils import get_tracking_uri

    assert get_tracking_uri() == "http://my-mlflow:9999"


def test_get_or_create_experiment_creates_when_absent():
    from shelfsense.tracking.mlflow_utils import get_or_create_experiment

    with patch("shelfsense.tracking.mlflow_utils.MlflowClient") as MockClient:
        client = MagicMock()
        MockClient.return_value = client
        client.get_experiment_by_name.return_value = None
        client.create_experiment.return_value = "42"

        eid = get_or_create_experiment("test-exp", "http://localhost:5000")

        assert eid == "42"
        client.create_experiment.assert_called_once_with("test-exp")


# ── MLflowResource (resources.py) ─────────────────────────────────────────────


def test_mlflow_resource_default_fields():
    from shelfsense.orchestration.resources import MLflowResource

    r = MLflowResource(tracking_uri="http://localhost:5000")
    assert r.tracking_uri == "http://localhost:5000"
    assert r.experiment_name == "shelfsense-m5"


def test_mlflow_resource_log_asset_run_returns_run_id():
    from shelfsense.orchestration.resources import MLflowResource

    r = MLflowResource(tracking_uri="http://localhost:5000")

    run_mock = MagicMock()
    run_mock.info.run_id = "abc123"

    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=run_mock)
    ctx.__exit__ = MagicMock(return_value=False)

    with (
        patch("shelfsense.orchestration.resources.mlflow") as mock_mlflow,
        patch(
            "shelfsense.orchestration.resources.get_or_create_experiment",
            return_value="1",
        ),
    ):
        mock_mlflow.start_run.return_value = ctx
        run_id = r.log_asset_run("my-run", metrics={"val_wrmsse": 0.5}, params={"lr": 0.01})
        assert run_id == "abc123"
        mock_mlflow.set_tracking_uri.assert_called_once_with("http://localhost:5000")


def test_mlflow_resource_log_metrics_to_run_calls_client():
    from shelfsense.orchestration.resources import MLflowResource

    r = MLflowResource(tracking_uri="http://localhost:5000")

    with patch("shelfsense.orchestration.resources.MlflowClient") as MockClient:
        client = MagicMock()
        MockClient.return_value = client
        r.log_metrics_to_run("run-xyz", {"metric_a": 1.0, "metric_b": 2.5})
        assert client.log_metric.call_count == 2


# ── mlflow_utils.log_run context manager ──────────────────────────────────────


def test_get_or_create_experiment_returns_existing_id():
    """When the experiment already exists, returns its experiment_id."""
    from shelfsense.tracking.mlflow_utils import get_or_create_experiment

    with patch("shelfsense.tracking.mlflow_utils.MlflowClient") as MockClient:
        client = MagicMock()
        MockClient.return_value = client
        existing_exp = MagicMock()
        existing_exp.experiment_id = "99"
        client.get_experiment_by_name.return_value = existing_exp

        eid = get_or_create_experiment("existing-exp", "http://localhost:5000")

        assert eid == "99"
        client.create_experiment.assert_not_called()


def test_log_run_context_manager_yields_active_run():
    """log_run sets the URI, creates an experiment, opens a run, logs metrics."""
    from shelfsense.tracking.mlflow_utils import log_run

    run_mock = MagicMock()
    run_mock.info.run_id = "run-123"

    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=run_mock)
    ctx.__exit__ = MagicMock(return_value=False)

    with (
        patch("shelfsense.tracking.mlflow_utils.mlflow") as mock_mlflow,
        patch(
            "shelfsense.tracking.mlflow_utils.get_or_create_experiment",
            return_value="42",
        ),
    ):
        mock_mlflow.start_run.return_value = ctx
        with log_run(
            "test-run",
            experiment_name="shelfsense-m5",
            params={"lr": 0.01},
            metrics={"loss": 0.5},
            tracking_uri="http://localhost:5000",
        ) as run:
            assert run.info.run_id == "run-123"

        mock_mlflow.set_tracking_uri.assert_called_once_with("http://localhost:5000")
        mock_mlflow.log_params.assert_called_once_with({"lr": 0.01})
        mock_mlflow.log_metrics.assert_called_once_with({"loss": 0.5})


def test_log_run_with_artifact_paths(tmp_path):
    """log_run logs artifact directories and files via mlflow.log_artifacts/log_artifact."""
    from shelfsense.tracking.mlflow_utils import log_run

    artifact_dir = tmp_path / "arts"
    artifact_dir.mkdir()
    artifact_file = tmp_path / "result.txt"
    artifact_file.write_text("done")

    run_mock = MagicMock()
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=run_mock)
    ctx.__exit__ = MagicMock(return_value=False)

    with (
        patch("shelfsense.tracking.mlflow_utils.mlflow") as mock_mlflow,
        patch(
            "shelfsense.tracking.mlflow_utils.get_or_create_experiment",
            return_value="1",
        ),
    ):
        mock_mlflow.start_run.return_value = ctx
        with log_run(
            "artifact-run",
            artifact_paths=[str(artifact_dir), str(artifact_file)],
            tracking_uri="http://localhost:5000",
        ):
            pass

        mock_mlflow.log_artifacts.assert_called_once_with(str(artifact_dir))
        mock_mlflow.log_artifact.assert_called_once_with(str(artifact_file))
