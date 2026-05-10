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
import time

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

from shelfsense.orchestration.resources import MLflowResource


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
    mlflow_resource: MLflowResource,
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

    try:
        mlflow_resource.log_asset_run(
            run_name="raw_validated",
            metrics={
                "raw_sales_rows":    float(len(raw_sales)),
                "raw_calendar_rows": float(len(raw_calendar)),
                "raw_prices_rows":   float(len(raw_prices)),
            },
            tags={"asset": "raw_validated", "stage": "data_validation"},
        )
    except Exception as exc:
        context.log.warning(f"MLflow logging skipped: {exc}")

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
def features(
    context,
    raw_validated: dict,
    mlflow_resource: MLflowResource,
) -> str:
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
    t0 = time.time()
    rows = feature_engineer(
        sales_df=sales_df,
        calendar_df=calendar_df,
        prices_df=prices_df,
        output_dir=output_dir,
        last_day=last_day,
        verbose=True,
    )
    elapsed = round(time.time() - t0, 2)
    context.log.info(f"feature_engineer: {rows:,} rows written to {output_dir!r}")

    try:
        mlflow_resource.log_asset_run(
            run_name="features",
            metrics={
                "total_rows":   float(rows),
                "n_series":     float(len(sales_df)),
                "build_time_s": elapsed,
            },
            tags={"asset": "features", "output_dir": output_dir},
        )
    except Exception as exc:
        context.log.warning(f"MLflow logging skipped: {exc}")

    return output_dir


@asset(
    description=(
        "Validate each per-store feature parquet against the Pandera feature schema. "
        "Returns the same dir path on success; raises dagster.Failure on any violation."
    ),
)
def features_validated(
    context,
    features: str,
    mlflow_resource: MLflowResource,
) -> str:
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

    try:
        mlflow_resource.log_asset_run(
            run_name="features_validated",
            metrics={"validated_parquet_count": float(len(parquets))},
            tags={"asset": "features_validated"},
        )
    except Exception as exc:
        context.log.warning(f"MLflow logging skipped: {exc}")

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

_TVP13_CFG = {
    "objective": "tweedie", "tvp": 1.3,
    "learning_rate": 0.025, "num_leaves": 64,
    "min_data_in_leaf": 100, "feature_fraction": 0.7,
    "bagging_fraction": 0.9, "lambda_l2": 0.1,
    "num_boost_round": 3000, "early_stopping_rounds": 75, "horizon": 28,
}


@asset(
    config_schema={
        "model_dir": Field(str,  default_value="data/models/tvp_1p3"),
        "raw_dir":   Field(str,  default_value="data/raw/m5-forecasting-accuracy"),
        "test_mode": Field(bool, default_value=False,
                           description="10 boost rounds + horizon=1 for fast integration tests."),
    },
    description=(
        "Train 28 direct-horizon LightGBM models with Tweedie loss (tvp=1.3). "
        "Production best: val WRMSSE 0.6860, private LB 0.5693. "
        "Returns dict with model_dir path and val_wrmsse."
    ),
)
def model_tvp_13(
    context,
    features_validated: str,
    mlflow_resource: MLflowResource,
) -> dict:
    from shelfsense.models.lightgbm.multihorizon import MultiHorizonTrainer, DEFAULT_FEATURE_COLS

    cfg = context.op_config
    test_mode = cfg["test_mode"]
    trainer = MultiHorizonTrainer(_TVP13_CFG)

    t0 = time.time()
    result = trainer.fit(
        features_dir=features_validated,
        model_dir=cfg["model_dir"],
        feature_cols=DEFAULT_FEATURE_COLS,
        raw_dir=cfg["raw_dir"],
        num_boost_round_override=10 if test_mode else None,
        horizon_override=1 if test_mode else None,
    )
    elapsed = round(time.time() - t0, 2)
    context.log.info(
        f"model_tvp_13: val_wrmsse={result['val_wrmsse']:.4f}  "
        f"n_series={result['n_series']:,}  elapsed={elapsed}s"
    )

    try:
        mlflow_resource.log_asset_run(
            run_name="model_tvp_13",
            metrics={"val_wrmsse": result["val_wrmsse"], "train_time_s": elapsed},
            params={"objective": "tweedie", "tvp": "1.3", "test_mode": str(test_mode)},
            tags={"asset": "model_tvp_13", "model_dir": cfg["model_dir"]},
        )
    except Exception as exc:
        context.log.warning(f"MLflow logging skipped: {exc}")

    return {"model_dir": result["model_dir"], "val_wrmsse": result["val_wrmsse"]}


_TVP17_CFG = {
    "objective": "tweedie", "tvp": 1.7,
    "learning_rate": 0.025, "num_leaves": 64,
    "min_data_in_leaf": 100, "feature_fraction": 0.7,
    "bagging_fraction": 0.9, "lambda_l2": 0.1,
    "num_boost_round": 3000, "early_stopping_rounds": 75, "horizon": 28,
}

_RMSE_CFG = {
    "objective": "regression", "metric": "rmse",
    "learning_rate": 0.025, "num_leaves": 64,
    "min_data_in_leaf": 100, "feature_fraction": 0.7,
    "bagging_fraction": 0.9, "lambda_l2": 0.1,
    "num_boost_round": 3000, "early_stopping_rounds": 75, "horizon": 28,
}

_SD_CFG = {
    "objective": "tweedie", "tweedie_variance_power": 1.3,
    "optuna_trials": 10,
    "lr_min": 0.01, "lr_max": 0.1,
    "num_leaves_min": 31, "num_leaves_max": 127,
    "min_data_in_leaf_min": 20, "min_data_in_leaf_max": 100,
    "feature_fraction_min": 0.5, "feature_fraction_max": 1.0,
    "bagging_fraction_min": 0.5, "bagging_fraction_max": 1.0,
    "num_boost_round": 3000, "early_stopping_rounds": 75,
    "seed": 42, "num_threads": 0, "history_days": 200,
}

_YLAGS_CFG = {
    "objective": "tweedie", "tvp": 1.3,
    "learning_rate": 0.025, "num_leaves": 64,
    "min_data_in_leaf": 100, "feature_fraction": 0.7,
    "bagging_fraction": 0.9, "lambda_l2": 0.1,
    "num_boost_round": 3000, "early_stopping_rounds": 75, "horizon": 28,
}

_MODEL_CONFIG_SCHEMA = {
    "model_dir": Field(str,  default_value=""),
    "raw_dir":   Field(str,  default_value="data/raw/m5-forecasting-accuracy"),
    "test_mode": Field(bool, default_value=False,
                       description="10 boost rounds + horizon=1 for fast integration tests."),
}


@asset(
    config_schema={**_MODEL_CONFIG_SCHEMA,
                   "model_dir": Field(str, default_value="data/models/tvp_1p7")},
    description=(
        "Train 28 direct-horizon LightGBM models with Tweedie loss (tvp=1.7). "
        "Spike-emphasis complement to tvp=1.3 for ensemble diversity. "
        "Returns dict with model_dir path and val_wrmsse."
    ),
)
def model_tvp_17(
    context,
    features_validated: str,
    mlflow_resource: MLflowResource,
) -> dict:
    from shelfsense.models.lightgbm.multihorizon import MultiHorizonTrainer, DEFAULT_FEATURE_COLS

    cfg = context.op_config
    test_mode = cfg["test_mode"]
    trainer = MultiHorizonTrainer(_TVP17_CFG)

    t0 = time.time()
    result = trainer.fit(
        features_dir=features_validated,
        model_dir=cfg["model_dir"],
        feature_cols=DEFAULT_FEATURE_COLS,
        raw_dir=cfg["raw_dir"],
        num_boost_round_override=10 if test_mode else None,
        horizon_override=1 if test_mode else None,
    )
    elapsed = round(time.time() - t0, 2)
    context.log.info(
        f"model_tvp_17: val_wrmsse={result['val_wrmsse']:.4f}  "
        f"n_series={result['n_series']:,}  elapsed={elapsed}s"
    )

    try:
        mlflow_resource.log_asset_run(
            run_name="model_tvp_17",
            metrics={"val_wrmsse": result["val_wrmsse"], "train_time_s": elapsed},
            params={"objective": "tweedie", "tvp": "1.7", "test_mode": str(test_mode)},
            tags={"asset": "model_tvp_17", "variant": "tvp_17",
                  "feature_set": "default", "objective": "tweedie"},
        )
    except Exception as exc:
        context.log.warning(f"MLflow logging skipped: {exc}")

    return {"model_dir": result["model_dir"], "val_wrmsse": result["val_wrmsse"]}


@asset(
    config_schema={**_MODEL_CONFIG_SCHEMA,
                   "model_dir": Field(str, default_value="data/models/rmse_mh")},
    description=(
        "Train 28 direct-horizon LightGBM models with RMSE objective. "
        "Ensemble diversity component -- different loss surface from Tweedie variants. "
        "Returns dict with model_dir path and val_wrmsse."
    ),
)
def model_rmse_mh(
    context,
    features_validated: str,
    mlflow_resource: MLflowResource,
) -> dict:
    from shelfsense.models.lightgbm.multihorizon import MultiHorizonTrainer, DEFAULT_FEATURE_COLS

    cfg = context.op_config
    test_mode = cfg["test_mode"]
    trainer = MultiHorizonTrainer(_RMSE_CFG)

    t0 = time.time()
    result = trainer.fit(
        features_dir=features_validated,
        model_dir=cfg["model_dir"],
        feature_cols=DEFAULT_FEATURE_COLS,
        raw_dir=cfg["raw_dir"],
        num_boost_round_override=10 if test_mode else None,
        horizon_override=1 if test_mode else None,
    )
    elapsed = round(time.time() - t0, 2)
    context.log.info(
        f"model_rmse_mh: val_wrmsse={result['val_wrmsse']:.4f}  "
        f"n_series={result['n_series']:,}  elapsed={elapsed}s"
    )

    try:
        mlflow_resource.log_asset_run(
            run_name="model_rmse_mh",
            metrics={"val_wrmsse": result["val_wrmsse"], "train_time_s": elapsed},
            params={"objective": "regression", "test_mode": str(test_mode)},
            tags={"asset": "model_rmse_mh", "variant": "rmse_mh",
                  "feature_set": "default", "objective": "regression"},
        )
    except Exception as exc:
        context.log.warning(f"MLflow logging skipped: {exc}")

    return {"model_dir": result["model_dir"], "val_wrmsse": result["val_wrmsse"]}


@asset(
    config_schema={
        "model_dir": Field(str,  default_value="data/models/store_dept"),
        "raw_dir":   Field(str,  default_value="data/raw/m5-forecasting-accuracy"),
        "test_mode": Field(bool, default_value=False,
                           description="2 slices + 10 boost rounds for fast integration tests."),
    },
    description=(
        "Train 70 LightGBM models (one per store x dept slice) with recursive 28-step forecast. "
        "Fine-grained granularity captures local demand patterns. "
        "Returns dict with model_dir path and val_wrmsse."
    ),
)
def model_store_dept(
    context,
    features_validated: str,
    mlflow_resource: MLflowResource,
) -> dict:
    from shelfsense.models.lightgbm.store_dept import StoreDeptTrainer, DEFAULT_FEATURE_COLS

    cfg = context.op_config
    test_mode = cfg["test_mode"]
    trainer = StoreDeptTrainer(_SD_CFG)

    test_slices = [("CA_1", "FOODS_1")] if test_mode else None

    t0 = time.time()
    result = trainer.fit(
        features_dir=features_validated,
        model_dir=cfg["model_dir"],
        feature_cols=DEFAULT_FEATURE_COLS,
        raw_dir=cfg["raw_dir"],
        num_boost_round_override=10 if test_mode else None,
        optuna_trials_override=2 if test_mode else None,
        slices_override=test_slices,
    )
    elapsed = round(time.time() - t0, 2)
    n_slices = result["n_slices"] + result["n_slices_cached"]
    context.log.info(
        f"model_store_dept: val_wrmsse={result['val_wrmsse']:.4f}  "
        f"slices={n_slices}  elapsed={elapsed}s"
    )

    try:
        mlflow_resource.log_asset_run(
            run_name="model_store_dept",
            metrics={"val_wrmsse": result["val_wrmsse"], "train_time_s": elapsed,
                     "n_slices": float(n_slices)},
            params={"objective": "tweedie", "tvp": "1.3", "test_mode": str(test_mode)},
            tags={"asset": "model_store_dept", "variant": "store_dept",
                  "feature_set": "default", "objective": "tweedie"},
        )
    except Exception as exc:
        context.log.warning(f"MLflow logging skipped: {exc}")

    return {"model_dir": result["model_dir"], "val_wrmsse": result["val_wrmsse"]}


@asset(
    config_schema={**_MODEL_CONFIG_SCHEMA,
                   "model_dir": Field(str, default_value="data/models/ylags")},
    description=(
        "Train 28 direct-horizon LightGBM models with annual lag features (lag_91/182/364). "
        "Same Tweedie tvp=1.3 params as model_tvp_13; feature set adds lag_91/182/364. "
        "Reads same feature parquets as model_tvp_13 — annual lags already present on disk. "
        "Returns dict with model_dir path and val_wrmsse."
    ),
)
def model_ylags(
    context,
    features_validated: str,
    mlflow_resource: MLflowResource,
) -> dict:
    from shelfsense.models.lightgbm.multihorizon import MultiHorizonTrainer, YLAGS_FEATURE_COLS

    cfg = context.op_config
    test_mode = cfg["test_mode"]
    trainer = MultiHorizonTrainer(_YLAGS_CFG)

    t0 = time.time()
    result = trainer.fit(
        features_dir=features_validated,
        model_dir=cfg["model_dir"],
        feature_cols=YLAGS_FEATURE_COLS,
        raw_dir=cfg["raw_dir"],
        num_boost_round_override=10 if test_mode else None,
        horizon_override=1 if test_mode else None,
    )
    elapsed = round(time.time() - t0, 2)
    context.log.info(
        f"model_ylags: val_wrmsse={result['val_wrmsse']:.4f}  "
        f"n_series={result['n_series']:,}  elapsed={elapsed}s"
    )

    try:
        mlflow_resource.log_asset_run(
            run_name="model_ylags",
            metrics={"val_wrmsse": result["val_wrmsse"], "train_time_s": elapsed},
            params={"objective": "tweedie", "tvp": "1.3", "test_mode": str(test_mode)},
            tags={"asset": "model_ylags", "variant": "ylags",
                  "feature_set": "ylags", "objective": "tweedie"},
        )
    except Exception as exc:
        context.log.warning(f"MLflow logging skipped: {exc}")

    return {"model_dir": result["model_dir"], "val_wrmsse": result["val_wrmsse"]}


# -- Model asset checks --------------------------------------------------------

def _check_mh_pkl_count(model_dir: str, expected: int = 28) -> AssetCheckResult:
    pkls = glob.glob(os.path.join(model_dir, "h_*.pkl"))
    n = len(pkls)
    return AssetCheckResult(
        passed=n == expected,
        description=f"pkl count: {n} (expected {expected})",
        metadata={"count": n, "model_dir": model_dir},
    )


def _check_val_wrmsse(val_wrmsse: float, lo: float = 0.5, hi: float = 1.5) -> AssetCheckResult:
    return AssetCheckResult(
        passed=lo < val_wrmsse < hi,
        description=f"val_wrmsse: {val_wrmsse:.4f} (expected {lo}–{hi})",
        metadata={"val_wrmsse": val_wrmsse},
    )


@asset_check(asset="model_tvp_13", description="model_tvp_13 must write exactly 28 h_*.pkl files.")
def check_model_tvp_13_pkl_count(model_tvp_13: dict) -> AssetCheckResult:
    return _check_mh_pkl_count(model_tvp_13["model_dir"])


@asset_check(asset="model_tvp_13", description="model_tvp_13 val_wrmsse must be in (0.5, 1.5).")
def check_model_tvp_13_val_wrmsse(model_tvp_13: dict) -> AssetCheckResult:
    return _check_val_wrmsse(model_tvp_13["val_wrmsse"])


@asset_check(asset="model_tvp_17", description="model_tvp_17 must write exactly 28 h_*.pkl files.")
def check_model_tvp_17_pkl_count(model_tvp_17: dict) -> AssetCheckResult:
    return _check_mh_pkl_count(model_tvp_17["model_dir"])


@asset_check(asset="model_tvp_17", description="model_tvp_17 val_wrmsse must be in (0.5, 1.5).")
def check_model_tvp_17_val_wrmsse(model_tvp_17: dict) -> AssetCheckResult:
    return _check_val_wrmsse(model_tvp_17["val_wrmsse"])


@asset_check(asset="model_rmse_mh", description="model_rmse_mh must write exactly 28 h_*.pkl files.")
def check_model_rmse_mh_pkl_count(model_rmse_mh: dict) -> AssetCheckResult:
    return _check_mh_pkl_count(model_rmse_mh["model_dir"])


@asset_check(asset="model_rmse_mh", description="model_rmse_mh val_wrmsse must be in (0.5, 1.5).")
def check_model_rmse_mh_val_wrmsse(model_rmse_mh: dict) -> AssetCheckResult:
    return _check_val_wrmsse(model_rmse_mh["val_wrmsse"])


@asset_check(
    asset="model_store_dept",
    description="model_store_dept must write between 1 and 70 lgbm_SD_*.pkl files.",
)
def check_model_store_dept_pkl_count(model_store_dept: dict) -> AssetCheckResult:
    model_dir = model_store_dept["model_dir"]
    pkls = glob.glob(os.path.join(model_dir, "lgbm_SD_*.pkl"))
    n = len(pkls)
    return AssetCheckResult(
        passed=1 <= n <= 70,
        description=f"pkl count: {n} (expected 1–70)",
        metadata={"count": n, "model_dir": model_dir},
    )


@asset_check(asset="model_store_dept", description="model_store_dept val_wrmsse must be in (0.5, 1.5).")
def check_model_store_dept_val_wrmsse(model_store_dept: dict) -> AssetCheckResult:
    return _check_val_wrmsse(model_store_dept["val_wrmsse"])


@asset_check(asset="model_ylags", description="model_ylags must write exactly 28 h_*.pkl files.")
def check_model_ylags_pkl_count(model_ylags: dict) -> AssetCheckResult:
    return _check_mh_pkl_count(model_ylags["model_dir"])


@asset_check(asset="model_ylags", description="model_ylags val_wrmsse must be in (0.5, 1.5).")
def check_model_ylags_val_wrmsse(model_ylags: dict) -> AssetCheckResult:
    return _check_val_wrmsse(model_ylags["val_wrmsse"])


# -- Predictions asset checks --------------------------------------------------

def _check_preds_n_series(preds: dict, variant: str) -> AssetCheckResult:
    n = preds["n_series"]
    return AssetCheckResult(
        passed=n > 0,
        description=f"{variant}: {n} series in eval parquet",
        metadata={"n_series": n, "eval_path": preds["eval_path"]},
    )


@asset_check(asset="predictions_tvp_13", description="predictions_tvp_13 must contain at least 1 series.")
def check_predictions_tvp_13(predictions_tvp_13: dict) -> AssetCheckResult:
    return _check_preds_n_series(predictions_tvp_13, "tvp_13")


@asset_check(asset="predictions_tvp_17", description="predictions_tvp_17 must contain at least 1 series.")
def check_predictions_tvp_17(predictions_tvp_17: dict) -> AssetCheckResult:
    return _check_preds_n_series(predictions_tvp_17, "tvp_17")


@asset_check(asset="predictions_rmse_mh", description="predictions_rmse_mh must contain at least 1 series.")
def check_predictions_rmse_mh(predictions_rmse_mh: dict) -> AssetCheckResult:
    return _check_preds_n_series(predictions_rmse_mh, "rmse_mh")


@asset_check(asset="predictions_store_dept", description="predictions_store_dept must contain at least 1 series.")
def check_predictions_store_dept(predictions_store_dept: dict) -> AssetCheckResult:
    return _check_preds_n_series(predictions_store_dept, "store_dept")


@asset_check(asset="predictions_ylags", description="predictions_ylags must contain at least 1 series.")
def check_predictions_ylags(predictions_ylags: dict) -> AssetCheckResult:
    return _check_preds_n_series(predictions_ylags, "ylags")


@asset_check(asset="ensemble", description="ensemble val_wrmsse must be in (0, 5.0).")
def check_ensemble_val_wrmsse(ensemble: dict) -> AssetCheckResult:
    wrmsse = ensemble["val_wrmsse"]
    return AssetCheckResult(
        passed=0.0 < wrmsse < 5.0,
        description=f"ensemble val_wrmsse: {wrmsse:.4f} (expected 0–5)",
        metadata={"val_wrmsse": wrmsse},
    )


@asset_check(
    asset="submission",
    description="submission must have 60,980 rows in production or > 0 rows in test_mode.",
)
def check_submission_row_count(submission: dict) -> AssetCheckResult:
    n = submission["n_rows"]
    is_test = submission["test_mode"]
    passed = (n == 60980) if not is_test else (n > 0)
    expected = "60,980" if not is_test else ">0 (test_mode)"
    return AssetCheckResult(
        passed=passed,
        description=f"n_rows={n:,} (expected {expected})",
        metadata={"n_rows": n, "test_mode": is_test},
    )


# -- Predictions ---------------------------------------------------------------

_EVAL_ORIGIN = 1941   # forecast d_1942-d_1969 (Kaggle evaluation period)
_VAL_ORIGIN  = 1913   # forecast d_1914-d_1941 (scoring against known actuals)
_FCOLS       = [f"F{h}" for h in range(1, 29)]

_PREDS_CONFIG = {
    "raw_dir":   Field(str,  default_value="data/raw/m5-forecasting-accuracy"),
    "test_mode": Field(bool, default_value=False),
}


def _pad_horizons(df: pd.DataFrame) -> None:
    """Fill F2..F28 with F1 when only h_01.pkl was trained (test_mode)."""
    if "F1" in df.columns:
        for h in range(2, 29):
            if f"F{h}" not in df.columns:
                df[f"F{h}"] = df["F1"]


def _write_preds(
    trainer, model_dir: str, features_validated: str,
    feature_cols: list, preds_dir: str, test_mode: bool,
    horizon_override,
) -> tuple:
    """Predict at both origins; pad in test_mode; write parquets. Returns (eval_path, val_path, n_series)."""
    eval_df = trainer.predict(
        model_dir=model_dir,
        features_dir=features_validated,
        forecast_origin_day=_EVAL_ORIGIN,
        feature_cols=feature_cols,
        horizon_override=horizon_override,
    )
    val_df = trainer.predict(
        model_dir=model_dir,
        features_dir=features_validated,
        forecast_origin_day=_VAL_ORIGIN,
        feature_cols=feature_cols,
        horizon_override=horizon_override,
    )
    if test_mode:
        _pad_horizons(eval_df)
        _pad_horizons(val_df)
    os.makedirs(preds_dir, exist_ok=True)
    eval_path = os.path.join(preds_dir, "eval.parquet")
    val_path  = os.path.join(preds_dir, "val.parquet")
    eval_df[["id"] + _FCOLS].to_parquet(eval_path, index=False)
    val_df[["id"] + _FCOLS].to_parquet(val_path,  index=False)
    return eval_path, val_path, len(eval_df)


@asset(
    config_schema={
        **_PREDS_CONFIG,
        "preds_dir": Field(str, default_value="data/predictions/tvp_1p3"),
    },
    description=(
        "Predict d_1942-d_1969 (eval) and d_1914-d_1941 (val) with the tvp=1.3 models. "
        "Writes eval.parquet and val.parquet with id + F1..F28. "
        "Returns dict with eval_path, val_path, n_series."
    ),
)
def predictions_tvp_13(
    context,
    model_tvp_13: dict,
    features_validated: str,
    mlflow_resource: MLflowResource,
) -> dict:
    from shelfsense.models.lightgbm.multihorizon import MultiHorizonTrainer, DEFAULT_FEATURE_COLS
    cfg = context.op_config
    test_mode = cfg["test_mode"]
    trainer = MultiHorizonTrainer(_TVP13_CFG)
    eval_path, val_path, n_series = _write_preds(
        trainer, model_tvp_13["model_dir"], features_validated,
        DEFAULT_FEATURE_COLS, cfg["preds_dir"], test_mode,
        1 if test_mode else None,
    )
    context.log.info(f"predictions_tvp_13: {n_series} series → {cfg['preds_dir']}")
    try:
        mlflow_resource.log_asset_run(
            run_name="predictions_tvp_13",
            metrics={"n_series": float(n_series), "upstream_val_wrmsse": model_tvp_13["val_wrmsse"]},
            params={"eval_origin": str(_EVAL_ORIGIN), "val_origin": str(_VAL_ORIGIN), "test_mode": str(test_mode)},
            tags={"asset": "predictions_tvp_13", "variant": "tvp_13"},
        )
    except Exception as exc:
        context.log.warning(f"MLflow logging skipped: {exc}")
    return {"eval_path": eval_path, "val_path": val_path, "n_series": n_series}


@asset(
    config_schema={
        **_PREDS_CONFIG,
        "preds_dir": Field(str, default_value="data/predictions/tvp_1p7"),
    },
    description=(
        "Predict d_1942-d_1969 (eval) and d_1914-d_1941 (val) with the tvp=1.7 models. "
        "Writes eval.parquet and val.parquet with id + F1..F28. "
        "Returns dict with eval_path, val_path, n_series."
    ),
)
def predictions_tvp_17(
    context,
    model_tvp_17: dict,
    features_validated: str,
    mlflow_resource: MLflowResource,
) -> dict:
    from shelfsense.models.lightgbm.multihorizon import MultiHorizonTrainer, DEFAULT_FEATURE_COLS
    cfg = context.op_config
    test_mode = cfg["test_mode"]
    trainer = MultiHorizonTrainer(_TVP17_CFG)
    eval_path, val_path, n_series = _write_preds(
        trainer, model_tvp_17["model_dir"], features_validated,
        DEFAULT_FEATURE_COLS, cfg["preds_dir"], test_mode,
        1 if test_mode else None,
    )
    context.log.info(f"predictions_tvp_17: {n_series} series → {cfg['preds_dir']}")
    try:
        mlflow_resource.log_asset_run(
            run_name="predictions_tvp_17",
            metrics={"n_series": float(n_series), "upstream_val_wrmsse": model_tvp_17["val_wrmsse"]},
            params={"eval_origin": str(_EVAL_ORIGIN), "val_origin": str(_VAL_ORIGIN), "test_mode": str(test_mode)},
            tags={"asset": "predictions_tvp_17", "variant": "tvp_17"},
        )
    except Exception as exc:
        context.log.warning(f"MLflow logging skipped: {exc}")
    return {"eval_path": eval_path, "val_path": val_path, "n_series": n_series}


@asset(
    config_schema={
        **_PREDS_CONFIG,
        "preds_dir": Field(str, default_value="data/predictions/rmse_mh"),
    },
    description=(
        "Predict d_1942-d_1969 (eval) and d_1914-d_1941 (val) with the RMSE multi-horizon models. "
        "Writes eval.parquet and val.parquet with id + F1..F28. "
        "Returns dict with eval_path, val_path, n_series."
    ),
)
def predictions_rmse_mh(
    context,
    model_rmse_mh: dict,
    features_validated: str,
    mlflow_resource: MLflowResource,
) -> dict:
    from shelfsense.models.lightgbm.multihorizon import MultiHorizonTrainer, DEFAULT_FEATURE_COLS
    cfg = context.op_config
    test_mode = cfg["test_mode"]
    trainer = MultiHorizonTrainer(_RMSE_CFG)
    eval_path, val_path, n_series = _write_preds(
        trainer, model_rmse_mh["model_dir"], features_validated,
        DEFAULT_FEATURE_COLS, cfg["preds_dir"], test_mode,
        1 if test_mode else None,
    )
    context.log.info(f"predictions_rmse_mh: {n_series} series → {cfg['preds_dir']}")
    try:
        mlflow_resource.log_asset_run(
            run_name="predictions_rmse_mh",
            metrics={"n_series": float(n_series), "upstream_val_wrmsse": model_rmse_mh["val_wrmsse"]},
            params={"eval_origin": str(_EVAL_ORIGIN), "val_origin": str(_VAL_ORIGIN), "test_mode": str(test_mode)},
            tags={"asset": "predictions_rmse_mh", "variant": "rmse_mh"},
        )
    except Exception as exc:
        context.log.warning(f"MLflow logging skipped: {exc}")
    return {"eval_path": eval_path, "val_path": val_path, "n_series": n_series}


@asset(
    config_schema={
        **_PREDS_CONFIG,
        "preds_dir": Field(str, default_value="data/predictions/store_dept"),
    },
    description=(
        "Recursive 28-step forecast for all 70 store×dept slices. "
        "Writes eval.parquet (d_1941 origin) and val.parquet (d_1913 origin). "
        "Returns dict with eval_path, val_path, n_series."
    ),
)
def predictions_store_dept(
    context,
    model_store_dept: dict,
    features_validated: str,
    mlflow_resource: MLflowResource,
) -> dict:
    import gc
    from shelfsense.models.lightgbm.store_dept import StoreDeptTrainer
    cfg = context.op_config
    test_mode = cfg["test_mode"]
    trainer = StoreDeptTrainer(_SD_CFG)
    slices = [("CA_1", "FOODS_1")] if test_mode else None
    os.makedirs(cfg["preds_dir"], exist_ok=True)

    # Eval preds: one CSV load (sales + prices + calendar) for the forward forecast
    eval_df = trainer.predict(
        model_dir=model_store_dept["model_dir"],
        raw_dir=cfg["raw_dir"],
        forecast_origin_day=_EVAL_ORIGIN,
        slices_override=slices,
    )
    gc.collect()  # free the large CSVs before loading val preds

    # Val preds: read cached val_preds stored inside each pkl — no CSV load needed
    val_df = trainer.val_preds_from_cache(
        model_dir=model_store_dept["model_dir"],
        slices_override=slices,
    )
    eval_path = os.path.join(cfg["preds_dir"], "eval.parquet")
    val_path  = os.path.join(cfg["preds_dir"], "val.parquet")
    eval_df[["id"] + _FCOLS].to_parquet(eval_path, index=False)
    val_df[["id"] + _FCOLS].to_parquet(val_path,  index=False)
    n_series = len(eval_df)
    context.log.info(f"predictions_store_dept: {n_series} series → {cfg['preds_dir']}")
    try:
        mlflow_resource.log_asset_run(
            run_name="predictions_store_dept",
            metrics={"n_series": float(n_series), "upstream_val_wrmsse": model_store_dept["val_wrmsse"]},
            params={"eval_origin": str(_EVAL_ORIGIN), "val_origin": str(_VAL_ORIGIN), "test_mode": str(test_mode)},
            tags={"asset": "predictions_store_dept", "variant": "store_dept"},
        )
    except Exception as exc:
        context.log.warning(f"MLflow logging skipped: {exc}")
    return {"eval_path": eval_path, "val_path": val_path, "n_series": n_series}


@asset(
    config_schema={
        **_PREDS_CONFIG,
        "preds_dir": Field(str, default_value="data/predictions/ylags"),
    },
    description=(
        "Predict d_1942-d_1969 (eval) and d_1914-d_1941 (val) with the annual-lag models. "
        "Uses YLAGS_FEATURE_COLS (adds lag_91/182/364). "
        "Writes eval.parquet and val.parquet with id + F1..F28. "
        "Returns dict with eval_path, val_path, n_series."
    ),
)
def predictions_ylags(
    context,
    model_ylags: dict,
    features_validated: str,
    mlflow_resource: MLflowResource,
) -> dict:
    from shelfsense.models.lightgbm.multihorizon import MultiHorizonTrainer, YLAGS_FEATURE_COLS
    cfg = context.op_config
    test_mode = cfg["test_mode"]
    trainer = MultiHorizonTrainer(_YLAGS_CFG)
    eval_path, val_path, n_series = _write_preds(
        trainer, model_ylags["model_dir"], features_validated,
        YLAGS_FEATURE_COLS, cfg["preds_dir"], test_mode,
        1 if test_mode else None,
    )
    context.log.info(f"predictions_ylags: {n_series} series → {cfg['preds_dir']}")
    try:
        mlflow_resource.log_asset_run(
            run_name="predictions_ylags",
            metrics={"n_series": float(n_series), "upstream_val_wrmsse": model_ylags["val_wrmsse"]},
            params={"eval_origin": str(_EVAL_ORIGIN), "val_origin": str(_VAL_ORIGIN), "test_mode": str(test_mode)},
            tags={"asset": "predictions_ylags", "variant": "ylags"},
        )
    except Exception as exc:
        context.log.warning(f"MLflow logging skipped: {exc}")
    return {"eval_path": eval_path, "val_path": val_path, "n_series": n_series}


# -- Ensemble ------------------------------------------------------------------

@asset(
    config_schema={
        "preds_dir": Field(str,  default_value="data/predictions/ensemble"),
        "raw_dir":   Field(str,  default_value="data/raw/m5-forecasting-accuracy"),
        "n_trials":  Field(int,  default_value=50,
                           description="Optuna trials for weight search (5 in test_mode)."),
        "test_mode": Field(bool, default_value=False,
                           description="Blend only tvp_13+tvp_17, 5 Optuna trials."),
    },
    description=(
        "Optuna weight search over convex combination of all 5 prediction variants. "
        "Objective: val WRMSSE from d_1913 origin vs actuals d_1914-d_1941. "
        "In test_mode blends tvp_13+tvp_17 with 5 trials. "
        "Returns dict with blended_eval_path, blended_val_path, weights, val_wrmsse."
    ),
)
def ensemble(
    context,
    predictions_tvp_13: dict,
    predictions_tvp_17: dict,
    predictions_rmse_mh: dict,
    predictions_store_dept: dict,
    predictions_ylags: dict,
    mlflow_resource: MLflowResource,
) -> dict:
    import numpy as np
    import optuna
    from shelfsense.evaluation.wrmsse import compute_wrmsse

    cfg = context.op_config
    test_mode = cfg["test_mode"]
    n_trials = 5 if test_mode else cfg["n_trials"]
    os.makedirs(cfg["preds_dir"], exist_ok=True)

    # In test_mode only blend the two MH variants (same 100-series test features)
    if test_mode:
        active = {"tvp_13": predictions_tvp_13, "tvp_17": predictions_tvp_17}
    else:
        active = {
            "tvp_13":     predictions_tvp_13,
            "tvp_17":     predictions_tvp_17,
            "rmse_mh":    predictions_rmse_mh,
            "store_dept": predictions_store_dept,
            "ylags":      predictions_ylags,
        }
    variant_names = list(active.keys())
    n_variants = len(variant_names)

    # Load val predictions; intersect series for safety
    val_indexed = [
        pd.read_parquet(active[v]["val_path"]).set_index("id")
        for v in variant_names
    ]
    series_ids = sorted(val_indexed[0].index.tolist())
    for df in val_indexed[1:]:
        series_ids = sorted(set(series_ids) & set(df.index.tolist()))
    val_preds = [
        df.reindex(series_ids)[_FCOLS].values.astype(np.float32)
        for df in val_indexed
    ]

    # Load raw CSVs for WRMSSE
    raw_dir = cfg["raw_dir"]
    sales_eval  = pd.read_csv(os.path.join(raw_dir, "sales_train_evaluation.csv"))
    prices_df   = pd.read_csv(os.path.join(raw_dir, "sell_prices.csv"))
    calendar_df = pd.read_csv(os.path.join(raw_dir, "calendar.csv"))

    actual_cols = [f"d_{_VAL_ORIGIN + h}" for h in range(1, 29)]
    sub_sales = (
        sales_eval[sales_eval["id"].isin(series_ids)]
        .set_index("id").reindex(series_ids).reset_index()
    )
    actuals = sub_sales[actual_cols].values.astype(np.float32)

    # Optuna: minimise blended val WRMSSE
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def _obj(trial: optuna.Trial) -> float:
        raw_w = [trial.suggest_float(f"w_{v}", 0.0, 1.0) for v in variant_names]
        total = sum(raw_w)
        if total < 1e-9:
            return float("inf")
        w = [x / total for x in raw_w]
        blended = np.zeros((len(series_ids), 28), dtype=np.float64)
        for wi, vp in zip(w, val_preds):
            blended += wi * vp.astype(np.float64)
        blended = np.clip(blended, 0.0, None).astype(np.float32)
        try:
            score, _ = compute_wrmsse(blended, actuals, sub_sales, prices_df, calendar_df, _VAL_ORIGIN)
            return score
        except Exception:
            return float("inf")

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(_obj, n_trials=n_trials, show_progress_bar=False)

    raw_best = [study.best_params[f"w_{v}"] for v in variant_names]
    total = sum(raw_best)
    best_w = [x / total for x in raw_best]
    weights_dict = {v: round(w, 4) for v, w in zip(variant_names, best_w)}
    val_wrmsse = float(study.best_value)
    context.log.info(f"ensemble: weights={weights_dict}  val_wrmsse={val_wrmsse:.4f}")

    # Apply weights to eval predictions
    eval_indexed = [
        pd.read_parquet(active[v]["eval_path"]).set_index("id")
        for v in variant_names
    ]
    eval_ids = sorted(eval_indexed[0].index.tolist())
    for df in eval_indexed[1:]:
        eval_ids = sorted(set(eval_ids) & set(df.index.tolist()))

    blended_eval = np.zeros((len(eval_ids), 28), dtype=np.float64)
    for wi, df in zip(best_w, eval_indexed):
        blended_eval += wi * df.reindex(eval_ids)[_FCOLS].values.astype(np.float64)
    blended_eval = np.clip(blended_eval, 0.0, None).astype(np.float32)

    blended_val = np.zeros((len(series_ids), 28), dtype=np.float64)
    for wi, vp in zip(best_w, val_preds):
        blended_val += wi * vp.astype(np.float64)
    blended_val = np.clip(blended_val, 0.0, None).astype(np.float32)

    eval_df_out = pd.DataFrame(blended_eval, columns=_FCOLS)
    eval_df_out.insert(0, "id", eval_ids)
    val_df_out = pd.DataFrame(blended_val, columns=_FCOLS)
    val_df_out.insert(0, "id", series_ids)

    blended_eval_path = os.path.join(cfg["preds_dir"], "ensemble_eval.parquet")
    blended_val_path  = os.path.join(cfg["preds_dir"], "ensemble_val.parquet")
    eval_df_out.to_parquet(blended_eval_path, index=False)
    val_df_out.to_parquet(blended_val_path,   index=False)

    try:
        mlflow_resource.log_asset_run(
            run_name="ensemble",
            metrics={"ensemble_val_wrmsse": val_wrmsse, "n_optuna_trials": float(n_trials)},
            params={
                **{f"weight_{k}": str(v) for k, v in weights_dict.items()},
                "test_mode": str(test_mode),
            },
            tags={"asset": "ensemble", "n_variants": str(n_variants)},
        )
    except Exception as exc:
        context.log.warning(f"MLflow logging skipped: {exc}")

    return {
        "blended_eval_path": blended_eval_path,
        "blended_val_path":  blended_val_path,
        "weights":           weights_dict,
        "val_wrmsse":        val_wrmsse,
    }


# -- Submission ----------------------------------------------------------------

@asset(
    config_schema={
        "submissions_dir": Field(str,  default_value="submissions"),
        "raw_dir":         Field(str,  default_value="data/raw/m5-forecasting-accuracy"),
        "kaggle_submit":   Field(bool, default_value=False,
                                 description="Upload to Kaggle via CLI when True (skipped in test_mode)."),
        "test_mode":       Field(bool, default_value=False),
    },
    description=(
        "Build Kaggle-format submission CSV (60,980 rows: 30,490 validation + 30,490 evaluation). "
        "Validates against submission_schema. "
        "Optionally pushes via kaggle CLI when kaggle_submit=True. "
        "Returns dict with path, n_rows, test_mode."
    ),
)
def submission(context, ensemble: dict, mlflow_resource: MLflowResource) -> dict:
    import datetime
    import pandera as pa
    from shelfsense.data.schemas import submission_schema

    cfg = context.op_config
    test_mode = cfg["test_mode"]
    os.makedirs(cfg["submissions_dir"], exist_ok=True)

    eval_df = pd.read_parquet(ensemble["blended_eval_path"])
    val_df  = pd.read_parquet(ensemble["blended_val_path"])

    val_rows = val_df.copy()
    val_rows["id"] = val_rows["id"].str.replace("_evaluation", "_validation", regex=False)

    sub_df = pd.concat(
        [val_rows[["id"] + _FCOLS].sort_values("id"),
         eval_df[["id"] + _FCOLS].sort_values("id")],
        ignore_index=True,
    )

    try:
        submission_schema.validate(sub_df, lazy=True)
        context.log.info(f"submission schema OK: {len(sub_df):,} rows")
    except pa.errors.SchemaErrors as exc:
        context.log.warning(f"submission schema issues: {len(exc.failure_cases)} failures")

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(cfg["submissions_dir"], f"{ts}_ensemble.csv")
    sub_df.to_csv(csv_path, index=False)

    n_rows = len(sub_df)
    context.log.info(f"submission: {n_rows:,} rows → {csv_path}")

    kaggle_submitted = False
    if cfg["kaggle_submit"] and not test_mode:
        import subprocess
        res = subprocess.run(
            ["kaggle", "competitions", "submit", "-c", "m5-forecasting-accuracy",
             "-f", csv_path, "-m", "shelfsense-ensemble"],
            capture_output=True, text=True,
        )
        if res.returncode == 0:
            kaggle_submitted = True
            context.log.info(f"Kaggle upload OK: {res.stdout.strip()}")
        else:
            context.log.warning(f"Kaggle upload failed: {res.stderr.strip()}")

    try:
        mlflow_resource.log_asset_run(
            run_name="submission",
            metrics={"n_rows": float(n_rows)},
            params={"csv_path": csv_path, "test_mode": str(test_mode),
                    "kaggle_submitted": str(kaggle_submitted)},
            tags={"asset": "submission"},
        )
    except Exception as exc:
        context.log.warning(f"MLflow logging skipped: {exc}")

    return {"path": csv_path, "n_rows": n_rows, "test_mode": test_mode}


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
        check_model_tvp_13_pkl_count,
        check_model_tvp_13_val_wrmsse,
        check_model_tvp_17_pkl_count,
        check_model_tvp_17_val_wrmsse,
        check_model_rmse_mh_pkl_count,
        check_model_rmse_mh_val_wrmsse,
        check_model_store_dept_pkl_count,
        check_model_store_dept_val_wrmsse,
        check_model_ylags_pkl_count,
        check_model_ylags_val_wrmsse,
        check_predictions_tvp_13,
        check_predictions_tvp_17,
        check_predictions_rmse_mh,
        check_predictions_store_dept,
        check_predictions_ylags,
        check_ensemble_val_wrmsse,
        check_submission_row_count,
    ],
    resources={
        "mlflow_resource": MLflowResource(
            tracking_uri=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
        ),
    },
)
