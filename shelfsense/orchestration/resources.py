"""Dagster resources for the ShelfSense M5 pipeline."""
from __future__ import annotations

from typing import Any

import mlflow
from dagster import ConfigurableResource
from mlflow.tracking import MlflowClient

from shelfsense.tracking.mlflow_utils import EXPERIMENT_NAME, get_or_create_experiment


class MLflowResource(ConfigurableResource):
    """Dagster resource for MLflow experiment tracking.

    tracking_uri defaults to http://localhost:5000 for direct (non-Docker) use.
    Inside docker compose, set MLFLOW_TRACKING_URI=http://mlflow:5000 or pass the
    uri explicitly via resource config so Dagster connects to the mlflow service.
    """

    tracking_uri: str = "http://localhost:5000"
    experiment_name: str = EXPERIMENT_NAME

    def _client(self) -> MlflowClient:
        return MlflowClient(tracking_uri=self.tracking_uri)

    def get_experiment(self, name: str | None = None) -> str:
        """Return experiment_id for name, creating it if absent."""
        return get_or_create_experiment(
            name or self.experiment_name, tracking_uri=self.tracking_uri
        )

    def log_asset_run(
        self,
        run_name: str,
        metrics: dict[str, float] | None = None,
        params: dict[str, Any] | None = None,
        tags: dict[str, str] | None = None,
    ) -> str:
        """Open an MLflow run, log metrics/params/tags, close it, return run_id."""
        mlflow.set_tracking_uri(self.tracking_uri)
        exp_id = self.get_experiment()
        with mlflow.start_run(
            experiment_id=exp_id, run_name=run_name, tags=tags
        ) as run:
            if params:
                mlflow.log_params(params)
            if metrics:
                mlflow.log_metrics(metrics)
            return run.info.run_id

    def log_metrics_to_run(self, run_id: str, metrics: dict[str, float]) -> None:
        """Append metrics to an already-open run by run_id."""
        client = self._client()
        for key, value in metrics.items():
            client.log_metric(run_id, key, value)
