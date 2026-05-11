"""ShelfSense CLI — entry point for all pipeline commands."""

from __future__ import annotations

import os
import subprocess
from typing import Optional

import typer

import shelfsense

app = typer.Typer(
    name="shelfsense",
    help="ShelfSense M5 forecasting pipeline CLI.",
    no_args_is_help=True,
)

# ── Sub-apps ──────────────────────────────────────────────────────────────────

data_app = typer.Typer(help="Data download and validation commands.")
app.add_typer(data_app, name="data")

features_app = typer.Typer(help="Feature engineering commands.")
app.add_typer(features_app, name="features")

train_app = typer.Typer(help="Model training commands.")
app.add_typer(train_app, name="train")


# ── Constants ─────────────────────────────────────────────────────────────────

_RAW_DIR = "data/raw/m5-forecasting-accuracy"
_FEATURES_DIR = "data/processed/features"

_EXPECTED_RAW_FILES = [
    "sales_train_evaluation.csv",
    "sell_prices.csv",
    "calendar.csv",
    "sample_submission.csv",
]

# Model dirs indexed by asset name (test_mode vs production)
_MODEL_DIRS: dict[bool, dict[str, str]] = {
    False: {
        "model_tvp_13": "data/models/tvp_1p3",
        "model_tvp_17": "data/models/tvp_1p7",
        "model_rmse_mh": "data/models/rmse_mh",
        "model_store_dept": "data/models/store_dept",
        "model_ylags": "data/models/ylags",
        "model_per_store": "data/models/per_store",
        "model_per_dept": "data/models/per_dept",
    },
    True: {
        "model_tvp_13": "data/models/test_tvp_1p3",
        "model_tvp_17": "data/models/test_tvp_1p7",
        "model_rmse_mh": "data/models/test_rmse_mh",
        "model_store_dept": "data/models/test_store_dept",
        "model_ylags": "data/models/test_ylags",
        "model_per_store": "data/models/test_per_store",
        "model_per_dept": "data/models/test_per_dept",
    },
}

_PREDS_DIRS: dict[bool, dict[str, str]] = {
    False: {
        "predictions_tvp_13": "data/predictions/tvp_1p3",
        "predictions_tvp_17": "data/predictions/tvp_1p7",
        "predictions_rmse_mh": "data/predictions/rmse_mh",
        "predictions_store_dept": "data/predictions/store_dept",
        "predictions_ylags": "data/predictions/ylags",
        "predictions_per_store": "data/predictions/per_store",
        "predictions_per_dept": "data/predictions/per_dept",
        "ensemble": "data/predictions/ensemble",
    },
    True: {
        "predictions_tvp_13": "data/predictions/test_tvp_1p3",
        "predictions_tvp_17": "data/predictions/test_tvp_1p7",
        "predictions_rmse_mh": "data/predictions/test_rmse_mh",
        "predictions_store_dept": "data/predictions/test_store_dept",
        "predictions_ylags": "data/predictions/test_ylags",
        "predictions_per_store": "data/predictions/test_per_store",
        "predictions_per_dept": "data/predictions/test_per_dept",
        "ensemble": "data/predictions/test_ensemble",
    },
}

# All 22 asset names in topological order (raw → features → models → predictions → output).
_ASSET_NAMES: list[str] = [
    "raw_sales",
    "raw_calendar",
    "raw_prices",
    "raw_validated",
    "features",
    "features_validated",
    "model_tvp_13",
    "model_tvp_17",
    "model_rmse_mh",
    "model_store_dept",
    "model_ylags",
    "model_per_store",
    "model_per_dept",
    "predictions_tvp_13",
    "predictions_tvp_17",
    "predictions_rmse_mh",
    "predictions_store_dept",
    "predictions_ylags",
    "predictions_per_store",
    "predictions_per_dept",
    "ensemble",
    "submission",
]


# ── Helpers ───────────────────────────────────────────────────────────────────


def _mlflow_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")


def _is_test_mode() -> bool:
    return os.environ.get("SHELFSENSE_TEST_MODE", "0") == "1"


def _dag_run(assets: list, run_config: dict) -> bool:
    """Materialize assets, return True on success."""
    from dagster import materialize

    from shelfsense.orchestration.resources import MLflowResource

    result = materialize(
        assets=assets,
        resources={"mlflow_resource": MLflowResource(tracking_uri=_mlflow_uri())},
        run_config=run_config,
    )
    return result.success


def _raw_ops(raw_dir: str) -> dict:
    return {
        "raw_sales": {"config": {"raw_dir": raw_dir}},
        "raw_calendar": {"config": {"raw_dir": raw_dir}},
        "raw_prices": {"config": {"raw_dir": raw_dir}},
    }


def _features_op(output_dir: str, test_mode: bool) -> dict:
    cfg: dict = {"output_dir": output_dir, "last_day": 1941}
    if test_mode:
        cfg.update({"test_mode": True, "test_n_series": 100, "test_seed": 42})
    return {"features": {"config": cfg}}


def _full_ops_cfg(test_mode: bool, raw_dir: str = _RAW_DIR) -> dict:
    """Build the complete ops run_config dict for the full pipeline through ensemble."""
    ops: dict = {}
    ops.update(_raw_ops(raw_dir))
    ops.update(_features_op(_FEATURES_DIR, test_mode))

    mdirs = _MODEL_DIRS[test_mode]
    for key, model_dir in mdirs.items():
        ops[key] = {"config": {"model_dir": model_dir, "raw_dir": raw_dir, "test_mode": test_mode}}

    pdirs = _PREDS_DIRS[test_mode]
    for key in (
        "predictions_tvp_13",
        "predictions_tvp_17",
        "predictions_rmse_mh",
        "predictions_store_dept",
        "predictions_ylags",
        "predictions_per_store",
        "predictions_per_dept",
    ):
        ops[key] = {"config": {"preds_dir": pdirs[key], "raw_dir": raw_dir, "test_mode": test_mode}}

    ops["ensemble"] = {
        "config": {"preds_dir": pdirs["ensemble"], "raw_dir": raw_dir, "test_mode": test_mode}
    }
    return ops


# ── shelfsense version ────────────────────────────────────────────────────────


@app.command("version")
def version_cmd() -> None:
    """Print the installed shelfsense package version."""
    typer.echo(shelfsense.__version__)


# ── shelfsense data ───────────────────────────────────────────────────────────


@data_app.command("download")
def data_download(
    raw_dir: str = typer.Option(
        _RAW_DIR,
        "--raw-dir",
        help="Destination directory for the downloaded CSVs.",
    ),
) -> None:
    """Download the M5 competition CSVs from Kaggle into data/raw/."""
    os.makedirs(raw_dir, exist_ok=True)
    typer.echo(f"Downloading m5-forecasting-accuracy → {raw_dir} ...")
    r = subprocess.run(
        [
            "kaggle",
            "competitions",
            "download",
            "-c",
            "m5-forecasting-accuracy",
            "-p",
            raw_dir,
            "--unzip",
        ],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        typer.echo(r.stderr.strip() or r.stdout.strip(), err=True)
        raise typer.Exit(code=1)

    missing = [f for f in _EXPECTED_RAW_FILES if not os.path.exists(os.path.join(raw_dir, f))]
    if missing:
        typer.echo(f"Download incomplete — expected files missing: {missing}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"✓ {len(_EXPECTED_RAW_FILES)} files ready at {raw_dir}/")


@data_app.command("validate")
def data_validate(
    raw_dir: Optional[str] = typer.Option(
        None, "--raw-dir", help="Override path to raw CSV directory."
    ),
    features_dir: Optional[str] = typer.Option(
        None, "--features-dir", help="Override path to feature parquet directory."
    ),
) -> None:
    """Materialize raw_validated + features_validated Dagster assets."""
    from shelfsense.orchestration.assets import (
        features,
        features_validated,
        raw_calendar,
        raw_prices,
        raw_sales,
        raw_validated,
    )

    _raw = raw_dir or _RAW_DIR
    _feat = features_dir or _FEATURES_DIR
    _test = _is_test_mode()

    run_config = {
        "ops": {
            **_raw_ops(_raw),
            **_features_op(_feat, _test),
        }
    }

    typer.echo("Materializing raw_validated + features_validated ...")
    ok = _dag_run(
        [raw_sales, raw_calendar, raw_prices, raw_validated, features, features_validated],
        run_config,
    )
    if ok:
        typer.echo(f"✓ Validation passed  |  MLflow: {_mlflow_uri()}")
    else:
        typer.echo("✗ Validation failed", err=True)
        raise typer.Exit(code=1)


# ── shelfsense features ───────────────────────────────────────────────────────


@features_app.command("build")
def features_build(
    config_name: str = typer.Option(
        "features/default",
        "--config-name",
        help="Hydra config variant (e.g. features/default). Dagster path uses output_dir directly.",
    ),
    output_dir: Optional[str] = typer.Option(
        None,
        "--output-dir",
        help="Override output directory. Defaults to data/processed/features.",
    ),
) -> None:
    """Materialize features Dagster asset (writes per-store snappy parquets)."""
    from shelfsense.orchestration.assets import (
        features,
        raw_calendar,
        raw_prices,
        raw_sales,
        raw_validated,
    )

    _feat = output_dir or _FEATURES_DIR
    _test = _is_test_mode()

    if config_name != "features/default":
        typer.echo(
            f"Note: --config-name={config_name!r} — Dagster path uses output_dir={_feat!r}.",
            err=True,
        )

    run_config = {
        "ops": {
            **_raw_ops(_RAW_DIR),
            **_features_op(_feat, _test),
        }
    }

    typer.echo(f"Materializing features → {_feat} ...")
    ok = _dag_run(
        [raw_sales, raw_calendar, raw_prices, raw_validated, features],
        run_config,
    )
    if ok:
        suffix = "_test" if _test else ""
        typer.echo(f"✓ Features written to {_feat}{suffix}/  |  MLflow: {_mlflow_uri()}")
    else:
        typer.echo("✗ Feature build failed", err=True)
        raise typer.Exit(code=1)


# ── shelfsense train ──────────────────────────────────────────────────────────


@train_app.command("tweedie-mh")
def train_tweedie_mh(
    tvp: float = typer.Option(
        1.3,
        "--tvp",
        help="Tweedie variance power. Supported values: 1.3 (model_tvp_13), 1.7 (model_tvp_17).",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="Random seed (logged to MLflow; trainer seed is fixed at 42 in this version).",
    ),
) -> None:
    """Materialize model_tvp_13 or model_tvp_17 Dagster asset."""
    from shelfsense.orchestration.assets import (
        features,
        features_validated,
        model_tvp_13,
        model_tvp_17,
        raw_calendar,
        raw_prices,
        raw_sales,
        raw_validated,
    )

    _test = _is_test_mode()
    mdirs = _MODEL_DIRS[_test]

    if tvp == 1.3:
        model_asset = model_tvp_13
        asset_name = "model_tvp_13"
    elif tvp == 1.7:
        model_asset = model_tvp_17
        asset_name = "model_tvp_17"
    else:
        typer.echo(f"Error: --tvp must be 1.3 or 1.7 (got {tvp}).", err=True)
        raise typer.Exit(code=1)

    run_config = {
        "ops": {
            **_raw_ops(_RAW_DIR),
            **_features_op(_FEATURES_DIR, _test),
            asset_name: {
                "config": {
                    "model_dir": mdirs[asset_name],
                    "raw_dir": _RAW_DIR,
                    "test_mode": _test,
                    "seed": seed,
                }
            },
        }
    }

    typer.echo(f"Materializing {asset_name} (tvp={tvp}) ...")
    ok = _dag_run(
        [
            raw_sales,
            raw_calendar,
            raw_prices,
            raw_validated,
            features,
            features_validated,
            model_asset,
        ],
        run_config,
    )
    if ok:
        typer.echo(f"✓ {asset_name} trained  |  MLflow: {_mlflow_uri()}")
    else:
        typer.echo(f"✗ {asset_name} training failed", err=True)
        raise typer.Exit(code=1)


@train_app.command("store-dept")
def train_store_dept(
    slices: str = typer.Option(
        "all",
        "--slices",
        help=(
            "Comma-separated store×dept slice keys (e.g. CA_1_FOODS_3,TX_2_HOBBIES_1), "
            "or 'all' to train every combination. Custom slice selection is not yet "
            "supported via Dagster; pass 'all' or use SHELFSENSE_TEST_MODE=1 for a fast run."
        ),
    ),
) -> None:
    """Materialize model_store_dept Dagster asset (70 per-slice LightGBM models)."""
    from shelfsense.orchestration.assets import (
        features,
        features_validated,
        model_store_dept,
        raw_calendar,
        raw_prices,
        raw_sales,
        raw_validated,
    )

    _test = _is_test_mode()
    mdirs = _MODEL_DIRS[_test]

    run_config = {
        "ops": {
            **_raw_ops(_RAW_DIR),
            **_features_op(_FEATURES_DIR, _test),
            "model_store_dept": {
                "config": {
                    "model_dir": mdirs["model_store_dept"],
                    "raw_dir": _RAW_DIR,
                    "test_mode": _test,
                    "slices": slices,
                }
            },
        }
    }

    typer.echo("Materializing model_store_dept ...")
    ok = _dag_run(
        [
            raw_sales,
            raw_calendar,
            raw_prices,
            raw_validated,
            features,
            features_validated,
            model_store_dept,
        ],
        run_config,
    )
    if ok:
        typer.echo(f"✓ model_store_dept trained  |  MLflow: {_mlflow_uri()}")
    else:
        typer.echo("✗ model_store_dept training failed", err=True)
        raise typer.Exit(code=1)


@train_app.command("per-store")
def train_per_store() -> None:
    """Materialize model_per_store Dagster asset (10 per-store LightGBM model sets)."""
    from shelfsense.orchestration.assets import (
        features,
        features_validated,
        model_per_store,
        raw_calendar,
        raw_prices,
        raw_sales,
        raw_validated,
    )

    _test = _is_test_mode()
    mdirs = _MODEL_DIRS[_test]

    run_config = {
        "ops": {
            **_raw_ops(_RAW_DIR),
            **_features_op(_FEATURES_DIR, _test),
            "model_per_store": {
                "config": {
                    "model_dir": mdirs["model_per_store"],
                    "raw_dir": _RAW_DIR,
                    "test_mode": _test,
                }
            },
        }
    }

    typer.echo("Materializing model_per_store ...")
    ok = _dag_run(
        [
            raw_sales,
            raw_calendar,
            raw_prices,
            raw_validated,
            features,
            features_validated,
            model_per_store,
        ],
        run_config,
    )
    if ok:
        typer.echo(f"✓ model_per_store trained  |  MLflow: {_mlflow_uri()}")
    else:
        typer.echo("✗ model_per_store training failed", err=True)
        raise typer.Exit(code=1)


@train_app.command("per-dept")
def train_per_dept() -> None:
    """Materialize model_per_dept Dagster asset (7 per-dept LightGBM model sets)."""
    from shelfsense.orchestration.assets import (
        features,
        features_validated,
        model_per_dept,
        raw_calendar,
        raw_prices,
        raw_sales,
        raw_validated,
    )

    _test = _is_test_mode()
    mdirs = _MODEL_DIRS[_test]

    run_config = {
        "ops": {
            **_raw_ops(_RAW_DIR),
            **_features_op(_FEATURES_DIR, _test),
            "model_per_dept": {
                "config": {
                    "model_dir": mdirs["model_per_dept"],
                    "raw_dir": _RAW_DIR,
                    "test_mode": _test,
                }
            },
        }
    }

    typer.echo("Materializing model_per_dept ...")
    ok = _dag_run(
        [
            raw_sales,
            raw_calendar,
            raw_prices,
            raw_validated,
            features,
            features_validated,
            model_per_dept,
        ],
        run_config,
    )
    if ok:
        typer.echo(f"✓ model_per_dept trained  |  MLflow: {_mlflow_uri()}")
    else:
        typer.echo("✗ model_per_dept training failed", err=True)
        raise typer.Exit(code=1)


# ── shelfsense ensemble ───────────────────────────────────────────────────────


@app.command("ensemble")
def ensemble_cmd(
    candidates: str = typer.Option(
        "tvp_13,store_dept",
        "--candidates",
        help="Comma-separated model variant keys to blend (Optuna selects weights automatically).",
    ),
    method: str = typer.Option(
        "optuna",
        "--method",
        help="Weight search method (only 'optuna' is supported via Dagster).",
    ),
) -> None:
    """Materialize ensemble Dagster asset (Optuna convex-weight search on val WRMSSE)."""
    from shelfsense.orchestration.assets import (
        ensemble,
        features,
        features_validated,
        model_per_dept,
        model_per_store,
        model_rmse_mh,
        model_store_dept,
        model_tvp_13,
        model_tvp_17,
        model_ylags,
        predictions_per_dept,
        predictions_per_store,
        predictions_rmse_mh,
        predictions_store_dept,
        predictions_tvp_13,
        predictions_tvp_17,
        predictions_ylags,
        raw_calendar,
        raw_prices,
        raw_sales,
        raw_validated,
    )

    _test = _is_test_mode()
    pdirs = _PREDS_DIRS[_test]

    if method != "optuna":
        typer.echo(f"Note: --method={method!r} — only 'optuna' is supported via Dagster.", err=True)

    ops = _full_ops_cfg(_test)
    run_config = {"ops": ops}

    typer.echo("Materializing ensemble (runs full pipeline: models → predictions → blend) ...")
    ok = _dag_run(
        [
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
            model_per_store,
            model_per_dept,
            predictions_tvp_13,
            predictions_tvp_17,
            predictions_rmse_mh,
            predictions_store_dept,
            predictions_ylags,
            predictions_per_store,
            predictions_per_dept,
            ensemble,
        ],
        run_config,
    )
    if ok:
        typer.echo(f"✓ Ensemble materialized → {pdirs['ensemble']}/  |  MLflow: {_mlflow_uri()}")
    else:
        typer.echo("✗ Ensemble failed", err=True)
        raise typer.Exit(code=1)


# ── shelfsense submit ─────────────────────────────────────────────────────────


@app.command("submit")
def submit(
    variant: str = typer.Option(
        "best",
        "--variant",
        help="Model variant key to submit, or 'best' to auto-select by val WRMSSE.",
    ),
    kaggle: bool = typer.Option(
        False,
        "--kaggle/--no-kaggle",
        help="Push the submission CSV to Kaggle via the kaggle CLI.",
    ),
) -> None:
    """Materialize submission Dagster asset and optionally push to Kaggle."""
    from shelfsense.orchestration.assets import (
        ensemble,
        features,
        features_validated,
        model_per_dept,
        model_per_store,
        model_rmse_mh,
        model_store_dept,
        model_tvp_13,
        model_tvp_17,
        model_ylags,
        predictions_per_dept,
        predictions_per_store,
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

    _test = _is_test_mode()
    submissions_dir = "submissions/test" if _test else "submissions"

    ops = _full_ops_cfg(_test)
    ops["submission"] = {
        "config": {
            "submissions_dir": submissions_dir,
            "raw_dir": _RAW_DIR,
            "kaggle_submit": kaggle and not _test,
            "test_mode": _test,
        }
    }
    run_config = {"ops": ops}

    kaggle_note = " (--kaggle flag active)" if kaggle and not _test else ""
    typer.echo(f"Materializing submission{kaggle_note} ...")
    ok = _dag_run(
        [
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
            model_per_store,
            model_per_dept,
            predictions_tvp_13,
            predictions_tvp_17,
            predictions_rmse_mh,
            predictions_store_dept,
            predictions_ylags,
            predictions_per_store,
            predictions_per_dept,
            ensemble,
            submission,
        ],
        run_config,
    )
    if ok:
        typer.echo(f"✓ Submission written → {submissions_dir}/  |  MLflow: {_mlflow_uri()}")
    else:
        typer.echo("✗ Submission failed", err=True)
        raise typer.Exit(code=1)


# ── shelfsense materialize ────────────────────────────────────────────────────


@app.command("materialize")
def materialize_cmd(
    asset: str = typer.Option(
        "*",
        "--asset",
        help=(
            "Asset name(s) to materialize. '*' materializes all 22 assets (full pipeline). "
            "Comma-separate multiple names, e.g. 'features,model_tvp_13'."
        ),
    ),
) -> None:
    """Materialize Dagster assets by name. '--asset *' runs the full pipeline end-to-end."""
    import shelfsense.orchestration.assets as _a

    _test = _is_test_mode()

    # Build the full run_config once — Dagster ignores configs for ops not in the asset list.
    ops = _full_ops_cfg(_test)
    ops["submission"] = {
        "config": {
            "submissions_dir": "submissions/test" if _test else "submissions",
            "raw_dir": _RAW_DIR,
            "kaggle_submit": False,
            "test_mode": _test,
        }
    }
    run_config = {"ops": ops}

    asset_map: dict[str, object] = {name: getattr(_a, name) for name in _ASSET_NAMES}

    if asset == "*":
        assets_to_run = list(asset_map.values())
        typer.echo(f"Materializing all {len(assets_to_run)} assets ...")
    else:
        names = [n.strip() for n in asset.split(",")]
        unknown = [n for n in names if n not in asset_map]
        if unknown:
            typer.echo(
                f"Unknown asset(s): {unknown}. Available: {sorted(asset_map)}",
                err=True,
            )
            raise typer.Exit(code=1)
        assets_to_run = [asset_map[n] for n in names]
        typer.echo(f"Materializing {', '.join(names)} ...")

    ok = _dag_run(assets_to_run, run_config)
    if ok:
        typer.echo(f"✓ Materialization complete  |  MLflow: {_mlflow_uri()}")
    else:
        typer.echo("✗ Materialization failed", err=True)
        raise typer.Exit(code=1)


# ── shelfsense report ─────────────────────────────────────────────────────────


@app.command("report")
def report(
    regenerate_charts: bool = typer.Option(
        False,
        "--regenerate-charts/--no-regenerate-charts",
        help="Re-render all portfolio charts from current model scores.",
    ),
) -> None:
    """Regenerate the leaderboard and optionally all portfolio charts (Stage 6)."""
    typer.echo("report generation is deferred to Stage 6.", err=True)
    raise typer.Exit(code=1)


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app()
