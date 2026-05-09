"""Dagster asset graph for the ShelfSense M5 pipeline.

Full pipeline:
  raw_sales, raw_calendar, raw_prices  (SourceAsset — DVC-tracked external data)
  → raw_validated
  → features → features_validated
  → model_tvp_13 / model_tvp_17 / model_rmse_mh / model_store_dept / model_ylags
  → predictions_<variant>
  → ensemble
  → submission
"""
from __future__ import annotations

from dagster import AssetKey, AssetSpec, Definitions, asset


# ── External data sources ─────────────────────────────────────────────────────

raw_sales = AssetSpec(
    key="raw_sales",
    description=(
        "sales_train_evaluation.csv — 30,490 M5 series × 1,941 days. "
        "DVC-tracked; path from cfg.data.raw_dir."
    ),
)

raw_calendar = AssetSpec(
    key="raw_calendar",
    description=(
        "calendar.csv — day-level calendar features and event flags. "
        "DVC-tracked; path from cfg.data.raw_dir."
    ),
)

raw_prices = AssetSpec(
    key="raw_prices",
    description=(
        "sell_prices.csv — weekly item prices per store. "
        "DVC-tracked; path from cfg.data.raw_dir."
    ),
)


# ── Data validation ───────────────────────────────────────────────────────────

@asset(
    deps=["raw_sales", "raw_calendar", "raw_prices"],
    description=(
        "Load all three raw M5 CSVs and validate against Pandera schemas. "
        "Returns a dict of validated DataFrames keyed by 'sales', 'calendar', 'prices'."
    ),
)
def raw_validated(context) -> dict:
    raise NotImplementedError("Wired in commit 24")


# ── Feature engineering ───────────────────────────────────────────────────────

@asset(
    description=(
        "Run feature_engineer over all 30,490 series. "
        "Writes one parquet per store to data/processed/features/. "
        "Returns the output directory path as a string."
    ),
)
def features(context, raw_validated: dict) -> str:
    raise NotImplementedError("Wired in commit 24")


@asset(
    description=(
        "Validate each per-store feature parquet against the Pandera feature schema. "
        "Passes through the features directory path unchanged if all checks pass."
    ),
)
def features_validated(context, features: str) -> str:
    raise NotImplementedError("Wired in commit 24")


# ── Model training ────────────────────────────────────────────────────────────

@asset(
    description=(
        "Train 28 direct-horizon LightGBM models with Tweedie loss (tvp=1.3). "
        "Production best: val WRMSSE 0.6860, private LB 0.5693. "
        "Returns dict with model_dir path and val_wrmsse."
    ),
)
def model_tvp_13(context, features_validated: str) -> dict:
    raise NotImplementedError("Wired in commit 26")


@asset(
    description=(
        "Train 28 direct-horizon LightGBM models with Tweedie loss (tvp=1.7). "
        "Spike-emphasis complement to tvp=1.3 for ensemble diversity. "
        "Returns dict with model_dir path and val_wrmsse."
    ),
)
def model_tvp_17(context, features_validated: str) -> dict:
    raise NotImplementedError("Wired in commit 26")


@asset(
    description=(
        "Train 28 direct-horizon LightGBM models with RMSE objective. "
        "Ensemble diversity component — different loss surface from Tweedie variants. "
        "Returns dict with model_dir path and val_wrmsse."
    ),
)
def model_rmse_mh(context, features_validated: str) -> dict:
    raise NotImplementedError("Wired in commit 26")


@asset(
    description=(
        "Train 70 LightGBM models (one per store×dept slice) with recursive 28-step forecast. "
        "Fine-grained granularity captures local demand patterns. "
        "Returns dict with model_dir path and val_wrmsse."
    ),
)
def model_store_dept(context, features_validated: str) -> dict:
    raise NotImplementedError("Wired in commit 26")


@asset(
    description=(
        "Train 28 direct-horizon LightGBM models with annual lag features (lag_91/182/364). "
        "Tests yearly seasonality signal against the tvp=1.3 baseline. "
        "Returns dict with model_dir path and val_wrmsse."
    ),
)
def model_ylags(context, features_validated: str) -> dict:
    raise NotImplementedError("Wired in commit 26")


# ── Predictions ───────────────────────────────────────────────────────────────

@asset(
    description="Run 28 tvp=1.3 models over the evaluation origin (d_1941). Returns parquet path.",
)
def predictions_tvp_13(context, model_tvp_13: dict, features_validated: str) -> str:
    raise NotImplementedError("Wired in commit 27")


@asset(
    description="Run 28 tvp=1.7 models over the evaluation origin (d_1941). Returns parquet path.",
)
def predictions_tvp_17(context, model_tvp_17: dict, features_validated: str) -> str:
    raise NotImplementedError("Wired in commit 27")


@asset(
    description="Run 28 RMSE-mh models over the evaluation origin (d_1941). Returns parquet path.",
)
def predictions_rmse_mh(context, model_rmse_mh: dict, features_validated: str) -> str:
    raise NotImplementedError("Wired in commit 27")


@asset(
    description="Recursive 28-step forecast across all 70 store×dept slices. Returns parquet path.",
)
def predictions_store_dept(context, model_store_dept: dict, features_validated: str) -> str:
    raise NotImplementedError("Wired in commit 27")


@asset(
    description="Run 28 ylags models over the evaluation origin (d_1941). Returns parquet path.",
)
def predictions_ylags(context, model_ylags: dict, features_validated: str) -> str:
    raise NotImplementedError("Wired in commit 27")


# ── Ensemble + submission ─────────────────────────────────────────────────────

@asset(
    description=(
        "50-trial Optuna search over convex combination weights for all 5 prediction variants. "
        "Objective: val WRMSSE from same origin (d_1913). "
        "Returns blended predictions parquet path."
    ),
)
def ensemble(
    context,
    predictions_tvp_13: str,
    predictions_tvp_17: str,
    predictions_rmse_mh: str,
    predictions_store_dept: str,
    predictions_ylags: str,
) -> str:
    raise NotImplementedError("Wired in commit 27")


@asset(
    description=(
        "Write Kaggle-format submission CSV (60,980 rows, F1-F28 columns). "
        "Optionally push via kaggle CLI when config field kaggle_submit=True."
    ),
)
def submission(context, ensemble: str) -> str:
    raise NotImplementedError("Wired in commit 27")


# ── Definitions ───────────────────────────────────────────────────────────────

defs = Definitions(
    assets=[
        # Source assets (DVC-tracked external data)
        raw_sales,
        raw_calendar,
        raw_prices,
        # Data pipeline
        raw_validated,
        features,
        features_validated,
        # Model training (5 parallel variants)
        model_tvp_13,
        model_tvp_17,
        model_rmse_mh,
        model_store_dept,
        model_ylags,
        # Predictions (one per model variant)
        predictions_tvp_13,
        predictions_tvp_17,
        predictions_rmse_mh,
        predictions_store_dept,
        predictions_ylags,
        # Ensemble + Kaggle output
        ensemble,
        submission,
    ]
)
