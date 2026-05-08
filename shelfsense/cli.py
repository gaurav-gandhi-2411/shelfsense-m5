"""ShelfSense CLI — entry point for all pipeline commands."""
from __future__ import annotations

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


# ── shelfsense version ────────────────────────────────────────────────────────

@app.command("version")
def version_cmd() -> None:
    """Print the installed shelfsense package version."""
    typer.echo(shelfsense.__version__)


# ── shelfsense data ───────────────────────────────────────────────────────────

@data_app.command("download")
def data_download() -> None:
    """Download the M5 competition CSVs from Kaggle into data/raw/."""
    typer.echo("shelfsense data download")
    raise NotImplementedError(
        "Wired in commit 9 — requires Kaggle API credentials and DVC remote."
    )


@data_app.command("validate")
def data_validate(
    raw_dir: Optional[str] = typer.Option(
        None, "--raw-dir", help="Override path to raw CSV directory."
    ),
    features_dir: Optional[str] = typer.Option(
        None, "--features-dir", help="Override path to feature parquet directory."
    ),
) -> None:
    """Run Pandera schema checks on raw CSVs and processed feature parquets."""
    import os
    from hydra import compose, initialize_config_dir

    from shelfsense.data.load import M5Dataset

    config_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "config")
    )
    with initialize_config_dir(
        config_dir=config_dir, job_name="data_validate", version_base=None
    ):
        cfg = compose(config_name="config")

    ds = M5Dataset(
        raw_dir=raw_dir or cfg.data.raw_dir,
        features_dir=features_dir or cfg.data.processed_dir,
        validate=True,
    )

    passed = 0
    failed = 0

    typer.echo("Validating raw CSVs ...")
    raw_results = ds.validate_raw()
    for fname, result in raw_results.items():
        if fname.endswith("__error"):
            continue
        status = "PASS" if result else "FAIL"
        typer.echo(f"  [{status}] {fname}")
        if result:
            passed += 1
        else:
            failed += 1
            err = raw_results.get(f"{fname}__error", "")
            if err:
                typer.echo(f"         {err[:200]}", err=True)

    typer.echo("Validating feature parquets ...")
    feat_results = ds.validate_features()
    for fname, result in feat_results.items():
        if fname.endswith("__error"):
            continue
        status = "PASS" if result else "FAIL"
        typer.echo(f"  [{status}] {fname}")
        if result:
            passed += 1
        else:
            failed += 1

    total = passed + failed
    typer.echo("")
    typer.echo(f"{passed}/{total} checks passed.")
    if failed:
        raise typer.Exit(code=1)
# ── shelfsense features ───────────────────────────────────────────────────────

@features_app.command("build")
def features_build(
    config_name: str = typer.Option(
        "features/default",
        "--config-name",
        help=(
            "Hydra config name for the feature set to build "
            "(e.g. features/default, features/ylags)."
        ),
    ),
    output_dir: Optional[str] = typer.Option(
        None,
        "--output-dir",
        help="Override output directory. Defaults to cfg.data.processed_dir.",
    ),
) -> None:
    """Build feature parquets from raw M5 CSVs using the specified Hydra config."""
    import os
    from hydra import compose, initialize_config_dir

    from shelfsense.features.pipeline import feature_engineer_from_config

    # Parse "features/default" -> ["features=default"]  (plain name -> no override)
    overrides = []
    if "/" in config_name:
        group, variant = config_name.split("/", 1)
        overrides.append(f"{group}={variant}")

    config_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "config")
    )
    with initialize_config_dir(
        config_dir=config_dir, job_name="features_build", version_base=None
    ):
        cfg = compose(config_name="config", overrides=overrides)

    feature_engineer_from_config(cfg, output_dir=output_dir or None)


# ── shelfsense train ──────────────────────────────────────────────────────────

@train_app.command("tweedie-mh")
def train_tweedie_mh(
    tvp: float = typer.Option(
        1.3,
        "--tvp",
        help=(
            "Tweedie variance power. Production value is 1.3. "
            "Higher values increase zero-inflation penalization."
        ),
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="Random seed — logged to MLflow for deterministic reproduction.",
    ),
) -> None:
    """Train 28 direct-horizon LightGBM models with Tweedie loss (production path)."""
    typer.echo(f"shelfsense train tweedie-mh --tvp {tvp} --seed {seed}")
    raise NotImplementedError("Wired in commit 9.")


@train_app.command("store-dept")
def train_store_dept(
    slices: str = typer.Option(
        "all",
        "--slices",
        help=(
            "Comma-separated store×dept slice keys (e.g. CA_1_FOODS_3,TX_2_HOBBIES_1), "
            "or 'all' to train every combination."
        ),
    ),
) -> None:
    """Train per-store×dept LightGBM models (ensemble diversity component)."""
    typer.echo(f"shelfsense train store-dept --slices {slices}")
    raise NotImplementedError("Wired in commit 9.")


@train_app.command("per-store")
def train_per_store() -> None:
    """Train one LightGBM model per Walmart store (10 stores × 1 model each)."""
    typer.echo("shelfsense train per-store")
    raise NotImplementedError("Wired in commit 9.")


@train_app.command("per-dept")
def train_per_dept() -> None:
    """Train one LightGBM model per M5 department (7 departments × 1 model each)."""
    typer.echo("shelfsense train per-dept")
    raise NotImplementedError("Wired in commit 9.")


# ── shelfsense ensemble ───────────────────────────────────────────────────────

@app.command("ensemble")
def ensemble(
    candidates: str = typer.Option(
        "tvp_13,store_dept",
        "--candidates",
        help="Comma-separated model variant keys to blend (e.g. tvp_13,store_dept,ylags_mh).",
    ),
    method: str = typer.Option(
        "optuna",
        "--method",
        help=(
            "Weight search method: 'optuna' (50-trial Bayesian search on val WRMSSE), "
            "'equal' (uniform weights), or 'fixed' (weights from config)."
        ),
    ),
) -> None:
    """Blend prediction CSVs from multiple model variants into an ensemble submission."""
    typer.echo(f"shelfsense ensemble --candidates {candidates} --method {method}")
    raise NotImplementedError("Wired in commit 9.")


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
    """Write the submission CSV for the chosen variant, optionally submitting to Kaggle."""
    flag = " --kaggle" if kaggle else ""
    typer.echo(f"shelfsense submit --variant {variant}{flag}")
    raise NotImplementedError("Wired in commit 9.")


# ── shelfsense report ─────────────────────────────────────────────────────────

@app.command("report")
def report(
    regenerate_charts: bool = typer.Option(
        False,
        "--regenerate-charts/--no-regenerate-charts",
        help="Re-render all portfolio charts from current model scores.",
    ),
) -> None:
    """Regenerate the leaderboard and optionally all portfolio charts."""
    flag = " --regenerate-charts" if regenerate_charts else ""
    typer.echo(f"shelfsense report{flag}")
    raise NotImplementedError("Wired in Stage 6.")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app()
