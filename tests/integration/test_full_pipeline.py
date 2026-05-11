"""End-to-end pipeline integration test: raw → submission in test_mode.

Gated by RUN_FULL_PIPELINE_TEST=1 — skip in CI by default.
Budget: < 12 minutes.

Run manually:
    RUN_FULL_PIPELINE_TEST=1 uv run --no-dev pytest tests/integration/test_full_pipeline.py -v

Note: The single-process dagster.materialize() call trains all 5 models back-to-back.
On first run this takes ~5-7 min. On subsequent runs (pkls cached) it takes ~3-4 min.
WSL2 memory cap of ~8 GB is tight; close other processes if it OOMs.
"""

from __future__ import annotations

import os

import pytest
import requests

TRACKING_URI = "http://localhost:5000"
RAW_DIR = "data/raw/m5-forecasting-accuracy"


def _mlflow_reachable() -> bool:
    try:
        r = requests.get(f"{TRACKING_URI}/health", timeout=2)
        return r.status_code == 200 and r.text.strip() == "OK"
    except Exception:
        return False


pytestmark = [
    pytest.mark.skipif(
        not _mlflow_reachable(),
        reason="MLflow not reachable at localhost:5000 — start with: docker compose up -d",
    ),
    pytest.mark.skipif(
        os.environ.get("RUN_FULL_PIPELINE_TEST") != "1",
        reason="Full pipeline test disabled — set RUN_FULL_PIPELINE_TEST=1 to enable",
    ),
]


def test_full_pipeline_test_mode():
    """Materialize raw → submission in test_mode: 100 series, 1 horizon, 5 Optuna trials."""
    import pandas as pd
    from dagster import materialize

    from shelfsense.orchestration.assets import (
        ensemble,
        features,
        features_validated,
        model_rmse_mh,
        model_store_dept,
        model_tvp_13,
        model_tvp_17,
        model_ylags,
        predictions_rmse_mh,
        predictions_store_dept,
        predictions_tvp_13,
        predictions_tvp_17,
        predictions_ylags,
        raw_calendar,
        raw_prices,
        raw_sales,
        raw_validated,
        submission,
    )
    from shelfsense.orchestration.resources import MLflowResource

    run_config = {
        "ops": {
            "raw_sales": {"config": {"raw_dir": RAW_DIR}},
            "raw_calendar": {"config": {"raw_dir": RAW_DIR}},
            "raw_prices": {"config": {"raw_dir": RAW_DIR}},
            "features": {
                "config": {
                    "output_dir": "data/processed/features",
                    "last_day": 1941,
                    "test_mode": True,
                    "test_n_series": 100,
                    "test_seed": 42,
                }
            },
            "model_tvp_13": {
                "config": {
                    "model_dir": "data/models/test_tvp_1p3",
                    "raw_dir": RAW_DIR,
                    "test_mode": True,
                }
            },
            "model_tvp_17": {
                "config": {
                    "model_dir": "data/models/test_tvp_1p7",
                    "raw_dir": RAW_DIR,
                    "test_mode": True,
                }
            },
            "model_rmse_mh": {
                "config": {
                    "model_dir": "data/models/test_rmse_mh",
                    "raw_dir": RAW_DIR,
                    "test_mode": True,
                }
            },
            "model_store_dept": {
                "config": {
                    "model_dir": "data/models/test_store_dept",
                    "raw_dir": RAW_DIR,
                    "test_mode": True,
                }
            },
            "model_ylags": {
                "config": {
                    "model_dir": "data/models/test_ylags",
                    "raw_dir": RAW_DIR,
                    "test_mode": True,
                }
            },
            "predictions_tvp_13": {
                "config": {
                    "preds_dir": "data/predictions/test_tvp_1p3",
                    "raw_dir": RAW_DIR,
                    "test_mode": True,
                }
            },
            "predictions_tvp_17": {
                "config": {
                    "preds_dir": "data/predictions/test_tvp_1p7",
                    "raw_dir": RAW_DIR,
                    "test_mode": True,
                }
            },
            "predictions_rmse_mh": {
                "config": {
                    "preds_dir": "data/predictions/test_rmse_mh",
                    "raw_dir": RAW_DIR,
                    "test_mode": True,
                }
            },
            "predictions_store_dept": {
                "config": {
                    "preds_dir": "data/predictions/test_store_dept",
                    "raw_dir": RAW_DIR,
                    "test_mode": True,
                }
            },
            "predictions_ylags": {
                "config": {
                    "preds_dir": "data/predictions/test_ylags",
                    "raw_dir": RAW_DIR,
                    "test_mode": True,
                }
            },
            "ensemble": {
                "config": {
                    "preds_dir": "data/predictions/test_ensemble",
                    "raw_dir": RAW_DIR,
                    "test_mode": True,
                }
            },
            "submission": {
                "config": {
                    "submissions_dir": "submissions/test",
                    "raw_dir": RAW_DIR,
                    "kaggle_submit": False,
                    "test_mode": True,
                }
            },
        }
    }

    all_assets = [
        raw_sales,
        raw_calendar,
        raw_prices,
        raw_validated,
        features,
        features_validated,
        model_tvp_13,
        model_tvp_17,
        model_rmse_mh,
        model_store_dept,
        model_ylags,
        predictions_tvp_13,
        predictions_tvp_17,
        predictions_rmse_mh,
        predictions_store_dept,
        predictions_ylags,
        ensemble,
        submission,
    ]

    result = materialize(
        assets=all_assets,
        resources={"mlflow_resource": MLflowResource(tracking_uri=TRACKING_URI)},
        run_config=run_config,
    )
    assert result.success, "Full pipeline materialization failed"

    # Submission assertions
    sub_output = result.output_for_node("submission")
    assert os.path.exists(sub_output["path"]), f"Submission CSV not found: {sub_output['path']}"
    assert sub_output["n_rows"] > 0

    sub_df = pd.read_csv(sub_output["path"])
    assert list(sub_df.columns[:1]) == ["id"]
    assert all(f"F{h}" in sub_df.columns for h in range(1, 29))
    assert sub_df["id"].str.contains("_validation").any()
    assert sub_df["id"].str.contains("_evaluation").any()

    # Ensemble assertions
    ens_output = result.output_for_node("ensemble")
    assert 0 < ens_output["val_wrmsse"] < 10.0
    assert "tvp_13" in ens_output["weights"]
    assert abs(sum(ens_output["weights"].values()) - 1.0) < 1e-4

    # MLflow: check key assets logged
    import mlflow

    mlflow.set_tracking_uri(TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name("shelfsense-m5")
    assert exp is not None
    for asset_name in ["model_tvp_13", "ensemble", "submission"]:
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string=f"tags.asset = '{asset_name}'",
            max_results=1,
        )
        assert len(runs) > 0, f"No MLflow run for {asset_name}"
