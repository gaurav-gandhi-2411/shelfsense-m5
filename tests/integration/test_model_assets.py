"""Integration tests: materialize each model/predictions/ensemble/submission asset in test_mode.

Full upstream chain for multi-horizon variants (tvp_13/17, rmse_mh, ylags):
  raw_sales/calendar/prices -> raw_validated -> features(100 series) ->
  features_validated -> model_<variant> -> predictions_<variant>

For model_store_dept: test_mode=True (100-series features_test), 1 slice CA_1xFOODS_1.

Ensemble and submission tests build on previously trained test models
(pkl files cached from prior model test runs).

Skip when MLflow is not reachable. Run after: docker compose up -d

Each model test < 90s. predictions/ensemble/submission tests < 120s each
(models already trained on first run, pkls reused on subsequent runs).
"""

from __future__ import annotations

import os

import pytest
import requests

TRACKING_URI = "http://localhost:5000"
RAW_DIR = "data/raw/m5-forecasting-accuracy"
FEATURES_DIR = "data/processed/features"


def _mlflow_reachable() -> bool:
    try:
        r = requests.get(f"{TRACKING_URI}/health", timeout=2)
        return r.status_code == 200 and r.text.strip() == "OK"
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_REAL_DATA_TESTS") != "1",
    reason=(
        "Real-data integration tests disabled — set RUN_REAL_DATA_TESTS=1 to enable. "
        "Requires M5 CSVs at data/raw/m5-forecasting-accuracy/ and MLflow at localhost:5000."
    ),
)

# Apply @_forked only when tests will actually run (RUN_REAL_DATA_TESTS=1).
# Avoids pytest-forked teardown corruption of session state for subsequent tests.
_forked = pytest.mark.forked if os.environ.get("RUN_REAL_DATA_TESTS") == "1" else lambda f: f


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
        "raw_sales": {"config": {"raw_dir": RAW_DIR}},
        "raw_calendar": {"config": {"raw_dir": RAW_DIR}},
        "raw_prices": {"config": {"raw_dir": RAW_DIR}},
        "features": {
            "config": {
                "output_dir": FEATURES_DIR,
                "last_day": 1941,
                "test_mode": test_mode,
                "test_n_series": 100,
                "test_seed": 42,
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


@_forked
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


@_forked
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


@_forked
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


@_forked
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


@_forked
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


# ---------------------------------------------------------------------------
# predictions_tvp_13
# ---------------------------------------------------------------------------


@_forked
def test_predictions_tvp_13_test_mode():
    """Predict eval+val parquets from the cached test tvp=1.3 model."""
    from dagster import materialize

    from shelfsense.orchestration.assets import model_tvp_13, predictions_tvp_13
    from shelfsense.orchestration.resources import MLflowResource

    model_dir = "data/models/test_tvp_1p3"
    preds_dir = "data/predictions/test_tvp_1p3"
    run_config = {
        "ops": {
            **_test_features_run_config(test_mode=True),
            "model_tvp_13": {
                "config": {"model_dir": model_dir, "raw_dir": RAW_DIR, "test_mode": True}
            },
            "predictions_tvp_13": {
                "config": {"preds_dir": preds_dir, "raw_dir": RAW_DIR, "test_mode": True}
            },
        }
    }

    result = materialize(
        assets=_upstream_assets() + [model_tvp_13, predictions_tvp_13],
        resources={"mlflow_resource": MLflowResource(tracking_uri=TRACKING_URI)},
        run_config=run_config,
    )
    assert result.success

    output = result.output_for_node("predictions_tvp_13")
    assert "eval_path" in output and "val_path" in output
    assert os.path.exists(output["eval_path"])
    assert os.path.exists(output["val_path"])
    assert output["n_series"] > 0

    import pandas as pd

    eval_df = pd.read_parquet(output["eval_path"])
    assert "id" in eval_df.columns
    assert all(f"F{h}" in eval_df.columns for h in range(1, 29))


# ---------------------------------------------------------------------------
# predictions_tvp_17
# ---------------------------------------------------------------------------


@_forked
def test_predictions_tvp_17_test_mode():
    """Predict eval+val parquets from the cached test tvp=1.7 model."""
    from dagster import materialize

    from shelfsense.orchestration.assets import model_tvp_17, predictions_tvp_17
    from shelfsense.orchestration.resources import MLflowResource

    model_dir = "data/models/test_tvp_1p7"
    preds_dir = "data/predictions/test_tvp_1p7"
    run_config = {
        "ops": {
            **_test_features_run_config(test_mode=True),
            "model_tvp_17": {
                "config": {"model_dir": model_dir, "raw_dir": RAW_DIR, "test_mode": True}
            },
            "predictions_tvp_17": {
                "config": {"preds_dir": preds_dir, "raw_dir": RAW_DIR, "test_mode": True}
            },
        }
    }

    result = materialize(
        assets=_upstream_assets() + [model_tvp_17, predictions_tvp_17],
        resources={"mlflow_resource": MLflowResource(tracking_uri=TRACKING_URI)},
        run_config=run_config,
    )
    assert result.success

    output = result.output_for_node("predictions_tvp_17")
    assert "eval_path" in output and "val_path" in output
    assert os.path.exists(output["eval_path"])
    assert output["n_series"] > 0


# ---------------------------------------------------------------------------
# predictions_rmse_mh
# ---------------------------------------------------------------------------


@_forked
def test_predictions_rmse_mh_test_mode():
    """Predict eval+val parquets from the cached test rmse_mh model."""
    from dagster import materialize

    from shelfsense.orchestration.assets import model_rmse_mh, predictions_rmse_mh
    from shelfsense.orchestration.resources import MLflowResource

    model_dir = "data/models/test_rmse_mh"
    preds_dir = "data/predictions/test_rmse_mh"
    run_config = {
        "ops": {
            **_test_features_run_config(test_mode=True),
            "model_rmse_mh": {
                "config": {"model_dir": model_dir, "raw_dir": RAW_DIR, "test_mode": True}
            },
            "predictions_rmse_mh": {
                "config": {"preds_dir": preds_dir, "raw_dir": RAW_DIR, "test_mode": True}
            },
        }
    }

    result = materialize(
        assets=_upstream_assets() + [model_rmse_mh, predictions_rmse_mh],
        resources={"mlflow_resource": MLflowResource(tracking_uri=TRACKING_URI)},
        run_config=run_config,
    )
    assert result.success

    output = result.output_for_node("predictions_rmse_mh")
    assert "eval_path" in output and "val_path" in output
    assert os.path.exists(output["eval_path"])
    assert output["n_series"] > 0


# ---------------------------------------------------------------------------
# predictions_store_dept
# ---------------------------------------------------------------------------


@_forked
def test_predictions_store_dept_test_mode():
    """Predict eval+val parquets for CA_1xFOODS_1 from the cached test model."""
    from dagster import materialize

    from shelfsense.orchestration.assets import model_store_dept, predictions_store_dept
    from shelfsense.orchestration.resources import MLflowResource

    model_dir = "data/models/test_store_dept"
    preds_dir = "data/predictions/test_store_dept"
    run_config = {
        "ops": {
            **_test_features_run_config(test_mode=True),
            "model_store_dept": {
                "config": {"model_dir": model_dir, "raw_dir": RAW_DIR, "test_mode": True}
            },
            "predictions_store_dept": {
                "config": {"preds_dir": preds_dir, "raw_dir": RAW_DIR, "test_mode": True}
            },
        }
    }

    result = materialize(
        assets=_upstream_assets() + [model_store_dept, predictions_store_dept],
        resources={"mlflow_resource": MLflowResource(tracking_uri=TRACKING_URI)},
        run_config=run_config,
    )
    assert result.success

    output = result.output_for_node("predictions_store_dept")
    assert "eval_path" in output and "val_path" in output
    assert os.path.exists(output["eval_path"])
    assert output["n_series"] > 0


# ---------------------------------------------------------------------------
# predictions_ylags
# ---------------------------------------------------------------------------


@_forked
def test_predictions_ylags_test_mode():
    """Predict eval+val parquets from the cached test ylags model."""
    from dagster import materialize

    from shelfsense.orchestration.assets import model_ylags, predictions_ylags
    from shelfsense.orchestration.resources import MLflowResource

    model_dir = "data/models/test_ylags"
    preds_dir = "data/predictions/test_ylags"
    run_config = {
        "ops": {
            **_test_features_run_config(test_mode=True),
            "model_ylags": {
                "config": {"model_dir": model_dir, "raw_dir": RAW_DIR, "test_mode": True}
            },
            "predictions_ylags": {
                "config": {"preds_dir": preds_dir, "raw_dir": RAW_DIR, "test_mode": True}
            },
        }
    }

    result = materialize(
        assets=_upstream_assets() + [model_ylags, predictions_ylags],
        resources={"mlflow_resource": MLflowResource(tracking_uri=TRACKING_URI)},
        run_config=run_config,
    )
    assert result.success

    output = result.output_for_node("predictions_ylags")
    assert "eval_path" in output and "val_path" in output
    assert os.path.exists(output["eval_path"])
    assert output["n_series"] > 0


# ---------------------------------------------------------------------------
# ensemble + submission (require all 5 models trained first)
# ---------------------------------------------------------------------------

_MODEL_DIRS = {
    "model_tvp_13": "data/models/test_tvp_1p3",
    "model_tvp_17": "data/models/test_tvp_1p7",
    "model_rmse_mh": "data/models/test_rmse_mh",
    "model_store_dept": "data/models/test_store_dept",
    "model_ylags": "data/models/test_ylags",
}
_PREDS_DIRS = {
    "predictions_tvp_13": "data/predictions/test_tvp_1p3",
    "predictions_tvp_17": "data/predictions/test_tvp_1p7",
    "predictions_rmse_mh": "data/predictions/test_rmse_mh",
    "predictions_store_dept": "data/predictions/test_store_dept",
    "predictions_ylags": "data/predictions/test_ylags",
    "ensemble": "data/predictions/test_ensemble",
}

_ensemble_test_enabled = pytest.mark.skipif(
    os.environ.get("RUN_ENSEMBLE_TEST") != "1",
    reason="Ensemble/submission tests disabled — set RUN_ENSEMBLE_TEST=1 to enable. "
    "Requires ~12GB RAM; materializes 17 assets in one process.",
)


def _all_model_assets():
    from shelfsense.orchestration.assets import (
        model_rmse_mh,
        model_store_dept,
        model_tvp_13,
        model_tvp_17,
        model_ylags,
    )

    return [model_tvp_13, model_tvp_17, model_rmse_mh, model_store_dept, model_ylags]


def _all_predictions_assets():
    from shelfsense.orchestration.assets import (
        predictions_rmse_mh,
        predictions_store_dept,
        predictions_tvp_13,
        predictions_tvp_17,
        predictions_ylags,
    )

    return [
        predictions_tvp_13,
        predictions_tvp_17,
        predictions_rmse_mh,
        predictions_store_dept,
        predictions_ylags,
    ]


def _full_run_config(include_submission: bool = False) -> dict:
    cfg = {
        "ops": {
            **_test_features_run_config(test_mode=True),
            "model_tvp_13": {
                "config": {
                    "model_dir": _MODEL_DIRS["model_tvp_13"],
                    "raw_dir": RAW_DIR,
                    "test_mode": True,
                }
            },
            "model_tvp_17": {
                "config": {
                    "model_dir": _MODEL_DIRS["model_tvp_17"],
                    "raw_dir": RAW_DIR,
                    "test_mode": True,
                }
            },
            "model_rmse_mh": {
                "config": {
                    "model_dir": _MODEL_DIRS["model_rmse_mh"],
                    "raw_dir": RAW_DIR,
                    "test_mode": True,
                }
            },
            "model_store_dept": {
                "config": {
                    "model_dir": _MODEL_DIRS["model_store_dept"],
                    "raw_dir": RAW_DIR,
                    "test_mode": True,
                }
            },
            "model_ylags": {
                "config": {
                    "model_dir": _MODEL_DIRS["model_ylags"],
                    "raw_dir": RAW_DIR,
                    "test_mode": True,
                }
            },
            "predictions_tvp_13": {
                "config": {
                    "preds_dir": _PREDS_DIRS["predictions_tvp_13"],
                    "raw_dir": RAW_DIR,
                    "test_mode": True,
                }
            },
            "predictions_tvp_17": {
                "config": {
                    "preds_dir": _PREDS_DIRS["predictions_tvp_17"],
                    "raw_dir": RAW_DIR,
                    "test_mode": True,
                }
            },
            "predictions_rmse_mh": {
                "config": {
                    "preds_dir": _PREDS_DIRS["predictions_rmse_mh"],
                    "raw_dir": RAW_DIR,
                    "test_mode": True,
                }
            },
            "predictions_store_dept": {
                "config": {
                    "preds_dir": _PREDS_DIRS["predictions_store_dept"],
                    "raw_dir": RAW_DIR,
                    "test_mode": True,
                }
            },
            "predictions_ylags": {
                "config": {
                    "preds_dir": _PREDS_DIRS["predictions_ylags"],
                    "raw_dir": RAW_DIR,
                    "test_mode": True,
                }
            },
            "ensemble": {
                "config": {
                    "preds_dir": _PREDS_DIRS["ensemble"],
                    "raw_dir": RAW_DIR,
                    "test_mode": True,
                }
            },
        }
    }
    if include_submission:
        cfg["ops"]["submission"] = {
            "config": {
                "submissions_dir": "submissions/test",
                "raw_dir": RAW_DIR,
                "kaggle_submit": False,
                "test_mode": True,
            }
        }
    return cfg


@_ensemble_test_enabled
@_forked
def test_ensemble_test_mode():
    """Ensemble: 5-trial Optuna over tvp_13+tvp_17 val preds (test_mode)."""
    from dagster import materialize

    from shelfsense.orchestration.assets import ensemble
    from shelfsense.orchestration.resources import MLflowResource

    result = materialize(
        assets=_upstream_assets() + _all_model_assets() + _all_predictions_assets() + [ensemble],
        resources={"mlflow_resource": MLflowResource(tracking_uri=TRACKING_URI)},
        run_config=_full_run_config(include_submission=False),
    )
    assert result.success

    output = result.output_for_node("ensemble")
    assert "blended_eval_path" in output
    assert "blended_val_path" in output
    assert os.path.exists(output["blended_eval_path"])
    assert os.path.exists(output["blended_val_path"])
    assert 0 < output["val_wrmsse"] < 10.0
    assert "tvp_13" in output["weights"]
    assert "tvp_17" in output["weights"]


@_ensemble_test_enabled
@_forked
def test_submission_test_mode():
    """Submission: build Kaggle-format CSV from ensemble predictions (test_mode)."""
    from dagster import materialize

    from shelfsense.orchestration.assets import ensemble, submission
    from shelfsense.orchestration.resources import MLflowResource

    result = materialize(
        assets=(
            _upstream_assets()
            + _all_model_assets()
            + _all_predictions_assets()
            + [ensemble, submission]
        ),
        resources={"mlflow_resource": MLflowResource(tracking_uri=TRACKING_URI)},
        run_config=_full_run_config(include_submission=True),
    )
    assert result.success

    output = result.output_for_node("submission")
    assert "path" in output
    assert os.path.exists(output["path"])
    assert output["n_rows"] > 0
    assert output["test_mode"] is True

    import pandas as pd

    sub_df = pd.read_csv(output["path"])
    assert "id" in sub_df.columns
    assert all(f"F{h}" in sub_df.columns for h in range(1, 29))
    assert sub_df["id"].str.contains("_validation").any()
    assert sub_df["id"].str.contains("_evaluation").any()
