"""Dagster asset graph for the ShelfSense M5 pipeline.

Full pipeline:
  raw_sales, raw_calendar, raw_prices  (load from DVC-tracked CSVs)
  -> raw_validated                      (Pandera schema checks)
  -> features                           (feature_engineer, per-store parquets)
  -> features_validated                 (feature schema checks)
  -> model_tvp_13 / model_tvp_17 / model_rmse_mh / model_store_dept / model_ylags
  -> predictions_<variant>
  -> ensemble
  -> submission
"""
from __future__ import annotations

import glob
import os

import pandas as pd
from dagster import (
    AssetCheckResult,
    AssetKey,
    Definitions,
    Failure,
    Field,
    asset,
    asset_check,
)


# -- Raw data loaders ----------------------------------------------------------

@asset(
    config_schema={"raw_dir": Field(str, default_value="data/raw/m5-forecasting-accuracy")},
    description=(
        "sales_train_evaluation.csv -- 30,490 M5 series x 1,941 days. "
        "DVC-tracked; default path matches cfg.data.raw_dir."
    ),
)
def raw_sales(context) -> pd.DataFrame:
    from shelfsense.data.load import M5Dataset
    ds = M5Dataset(raw_dir=context.op_config["raw_dir"], features_dir="", validate=False)
    df = ds.sales
    context.log.info(f"Loaded sales: {len(df):,} rows x {len(df.columns)} cols")
    return df


@asset(
    config_schema={"raw_dir": Field(str, default_value="data/raw/m5-forecasting-accuracy")},
    description=(
        "calendar.csv -- day-level calendar features and event flags. "
        "DVC-tracked; default path matches cfg.data.raw_dir."
    ),
)
def raw_calendar(context) -> pd.DataFrame:
    from shelfsense.data.load import M5Dataset
    ds = M5Dataset(raw_dir=context.op_config["raw_dir"], features_dir="", validate=False)
    df = ds.calendar
    context.log.info(f"Loaded calendar: {len(df):,} rows")
    return df


@asset(
    config_schema={"raw_dir": Field(str, default_value="data/raw/m5-forecasting-accuracy")},
    description=(
        "sell_prices.csv -- weekly item prices per store. "
        "DVC-tracked; default path matches cfg.data.raw_dir."
    ),
)
def raw_prices(context) -> pd.DataFrame:
    from shelfsense.data.load import M5Dataset
    ds = M5Dataset(raw_dir=context.op_config["raw_dir"], features_dir="", validate=False)
    df = ds.prices
    context.log.info(f"Loaded prices: {len(df):,} rows")
    return df


# -- Data validation -----------------------------------------------------------

@asset(
    description=(
        "Validate all three raw M5 DataFrames against their Pandera schemas. "
        "Returns dict keyed by 'sales', 'calendar', 'prices' with validated DataFrames. "
        "Raises dagster.Failure with violation details on schema errors."
    ),
)
def raw_validated(
    context,
    raw_sales: pd.DataFrame,
    raw_calendar: pd.DataFrame,
    raw_prices: pd.DataFrame,
) -> dict:
    import pandera as pa
    from shelfsense.data.schemas import (
        raw_calendar_schema,
        raw_prices_schema,
        raw_sales_schema,
    )

    validated: dict[str, pd.DataFrame] = {}
    for name, df, schema in [
        ("sales",    raw_sales,    raw_sales_schema),
        ("calendar", raw_calendar, raw_calendar_schema),
        ("prices",   raw_prices,   raw_prices_schema),
    ]:
        try:
            schema.validate(df, lazy=True)
            validated[name] = df
            context.log.info(f"[{name}] schema OK  ({len(df):,} rows)")
        except pa.errors.SchemaErrors as exc:
            n = len(exc.failure_cases)
            raise Failure(
                description=f"Pandera schema violation in '{name}': {n} failures",
                metadata={
                    "asset": name,
                    "n_violations": n,
                    "top_failures": str(exc.failure_cases.head(5).to_dict()),
                },
            )
    return validated


# -- Feature engineering -------------------------------------------------------

@asset(
    config_schema={
        "output_dir":    Field(str,  default_value="data/processed/features"),
        "last_day":      Field(int,  default_value=1941),
        "test_mode":     Field(bool, default_value=False,
                               description="Subset to test_n_series for fast integration tests."),
        "test_n_series": Field(int,  default_value=100,
                               description="Series to keep when test_mode=True."),
        "test_seed":     Field(int,  default_value=42,
                               description="RNG seed for series sampling in test_mode."),
    },
    description=(
        "Run feature_engineer() over all 30,490 series (or a subset in test_mode). "
        "Writes one snappy-compressed parquet per store to output_dir. "
        "Returns output_dir path; skips stores whose parquet already exists."
    ),
)
def features(context, raw_validated: dict) -> str:
    import numpy as np
    from shelfsense.features.pipeline import feature_engineer

    cfg = context.op_config
    output_dir = cfg["output_dir"]
    last_day = cfg["last_day"]
    sales_df = raw_validated["sales"].copy()
    calendar_df = raw_validated["calendar"]
    prices_df = raw_validated["prices"]

    if cfg["test_mode"]:
        rng = np.random.default_rng(cfg["test_seed"])
        n = min(cfg["test_n_series"], len(sales_df))
        idx = sorted(rng.choice(len(sales_df), size=n, replace=False).tolist())
        sales_df = sales_df.iloc[idx].reset_index(drop=True)
        output_dir = output_dir + "_test"
        context.log.info(f"test_mode: {n} series -> {output_dir!r}")

    os.makedirs(output_dir, exist_ok=True)
    rows = feature_engineer(
        sales_df=sales_df,
        calendar_df=calendar_df,
        prices_df=prices_df,
        output_dir=output_dir,
        last_day=last_day,
        verbose=True,
    )
    context.log.info(f"feature_engineer: {rows:,} rows written to {output_dir!r}")
    return output_dir


@asset(
    description=(
        "Validate each per-store feature parquet against the Pandera feature schema. "
        "Returns the same dir path on success; raises dagster.Failure on any violation."
    ),
)
def features_validated(context, features: str) -> str:
    import pandera as pa
    from shelfsense.data.schemas import feature_schema

    parquets = sorted(glob.glob(os.path.join(features, "*.parquet")))
    if not parquets:
        raise Failure(
            description=f"No parquets found in {features!r}",
            metadata={"output_dir": features},
        )
    for path in parquets:
        fname = os.path.basename(path)
        try:
            df = pd.read_parquet(path, engine="pyarrow")
            feature_schema.validate(df, lazy=True)
            context.log.info(f"[{fname}] feature schema OK  ({len(df):,} rows)")
        except pa.errors.SchemaErrors as exc:
            n = len(exc.failure_cases)
            raise Failure(
                description=f"Feature schema violation in {fname}: {n} failures",
                metadata={
                    "file": fname,
                    "n_violations": n,
                    "top_failures": str(exc.failure_cases.head(5).to_dict()),
                },
            )
    context.log.info(f"features_validated: {len(parquets)} parquets passed schema checks")
    return features


# -- Asset checks --------------------------------------------------------------

@asset_check(
    asset="raw_sales",
    description="sales_train_evaluation.csv must have exactly 30,490 rows (one per M5 series).",
)
def check_sales_row_count(raw_sales: pd.DataFrame) -> AssetCheckResult:
    n = len(raw_sales)
    return AssetCheckResult(
        passed=n == 30490,
        description=f"Row count: {n:,} (expected 30,490)",
        metadata={"n_rows": n},
    )


@asset_check(
    asset="features",
    description="Features dir must have exactly 10 store parquets (one per M5 store).",
)
def check_features_parquet_count(features: str) -> AssetCheckResult:
    parquets = glob.glob(os.path.join(features, "*.parquet"))
    n = len(parquets)
    return AssetCheckResult(
        passed=n == 10,
        description=f"Parquet count: {n} (expected 10)",
        metadata={"count": n, "output_dir": features},
    )


@asset_check(
    asset="features",
    description="d_num column must not contain NaN in any feature parquet.",
)
def check_features_no_nan_d_num(features: str) -> AssetCheckResult:
    for path in sorted(glob.glob(os.path.join(features, "*.parquet"))):
        fname = os.path.basename(path)
        df = pd.read_parquet(path, columns=["d_num"], engine="pyarrow")
        nan_count = int(df["d_num"].isna().sum())
        if nan_count > 0:
            return AssetCheckResult(
                passed=False,
                description=f"NaN in d_num: {fname} has {nan_count} NaN rows",
                metadata={"file": fname, "nan_count": nan_count},
            )
    return AssetCheckResult(
        passed=True,
        description="No NaN in d_num across all feature parquets",
    )


# -- Model training (wired in commit 26) --------------------------------------

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
        "Ensemble diversity component -- different loss surface from Tweedie variants. "
        "Returns dict with model_dir path and val_wrmsse."
    ),
)
def model_rmse_mh(context, features_validated: str) -> dict:
    raise NotImplementedError("Wired in commit 26")


@asset(
    description=(
        "Train 70 LightGBM models (one per store x dept slice) with recursive 28-step forecast. "
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


# -- Predictions (wired in commit 27) ------------------------------------------

@asset(description="Run 28 tvp=1.3 models over the evaluation origin (d_1941). Returns parquet path.")
def predictions_tvp_13(context, model_tvp_13: dict, features_validated: str) -> str:
    raise NotImplementedError("Wired in commit 27")


@asset(description="Run 28 tvp=1.7 models over the evaluation origin (d_1941). Returns parquet path.")
def predictions_tvp_17(context, model_tvp_17: dict, features_validated: str) -> str:
    raise NotImplementedError("Wired in commit 27")


@asset(description="Run 28 RMSE-mh models over the evaluation origin (d_1941). Returns parquet path.")
def predictions_rmse_mh(context, model_rmse_mh: dict, features_validated: str) -> str:
    raise NotImplementedError("Wired in commit 27")


@asset(description="Recursive 28-step forecast across all 70 store x dept slices. Returns parquet path.")
def predictions_store_dept(context, model_store_dept: dict, features_validated: str) -> str:
    raise NotImplementedError("Wired in commit 27")


@asset(description="Run 28 ylags models over the evaluation origin (d_1941). Returns parquet path.")
def predictions_ylags(context, model_ylags: dict, features_validated: str) -> str:
    raise NotImplementedError("Wired in commit 27")


# -- Ensemble + submission (wired in commit 27) --------------------------------

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


# -- Definitions ---------------------------------------------------------------

defs = Definitions(
    assets=[
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
    ],
    asset_checks=[
        check_sales_row_count,
        check_features_parquet_count,
        check_features_no_nan_d_num,
    ],
)
