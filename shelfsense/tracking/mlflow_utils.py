"""Standalone MLflow helpers — usable from any Python process, no Dagster required.

The Dagster resource (MLflowResource in orchestration/resources.py) wraps these
for use inside asset bodies. Call these directly for scripts or notebooks.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Generator
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = "shelfsense-m5"


def get_tracking_uri() -> str:
    """Return MLflow tracking URI from env MLFLOW_TRACKING_URI or localhost fallback."""
    return os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")


def get_or_create_experiment(name: str, tracking_uri: str | None = None) -> str:
    """Return experiment_id, creating the experiment if it does not exist yet."""
    uri = tracking_uri or get_tracking_uri()
    client = MlflowClient(tracking_uri=uri)
    exp = client.get_experiment_by_name(name)
    if exp is None:
        return client.create_experiment(name)
    return exp.experiment_id


@contextlib.contextmanager
def log_run(
    run_name: str,
    experiment_name: str = EXPERIMENT_NAME,
    params: dict[str, Any] | None = None,
    metrics: dict[str, float] | None = None,
    artifact_paths: list[str] | None = None,
    tags: dict[str, str] | None = None,
    tracking_uri: str | None = None,
) -> Generator[mlflow.ActiveRun, None, None]:
    """Context manager: open an MLflow run, log everything, close it on exit.

    artifact_paths: list of local file or directory paths to upload.
    Avoid large directories (e.g. feature parquets) — log output_dir as a tag instead.
    """
    uri = tracking_uri or get_tracking_uri()
    mlflow.set_tracking_uri(uri)
    experiment_id = get_or_create_experiment(experiment_name, tracking_uri=uri)
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name, tags=tags) as run:
        if params:
            mlflow.log_params(params)
        if metrics:
            mlflow.log_metrics(metrics)
        if artifact_paths:
            for path in artifact_paths:
                if os.path.isdir(path):
                    mlflow.log_artifacts(path)
                elif os.path.isfile(path):
                    mlflow.log_artifact(path)
        yield run
