"""Multi-horizon LightGBM trainer for direct 28-step forecasting.

Trains one model per horizon h in [1, 28].  Each model predicts sales h days
ahead; the target is shift(-h) per series.  All models share the same
hyperparameters (no per-horizon Optuna search).
"""

from __future__ import annotations

import gc
import os
import pickle
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from shelfsense.features.hierarchy import CAT_DTYPES

LAST_TRAIN = 1913
VAL_START = 1886
FEAT_START = 1000
HORIZON = 28

CAT_FEATURES = ["cat_id", "dept_id", "store_id", "state_id"]

DEFAULT_NUM_FEATURES = [
    "weekday",
    "month",
    "quarter",
    "year",
    "day_of_month",
    "week_of_year",
    "is_weekend",
    "is_holiday",
    "is_snap_ca",
    "is_snap_tx",
    "is_snap_wi",
    "days_since_event",
    "days_until_next_event",
    "sell_price",
    "price_change_pct",
    "price_relative_mean",
    "price_volatility",
    "has_price_change",
    "lag_7",
    "lag_14",
    "lag_28",
    "lag_56",
    "roll_mean_7",
    "roll_std_7",
    "roll_min_7",
    "roll_max_7",
    "roll_mean_28",
    "roll_std_28",
    "roll_min_28",
    "roll_max_28",
    "roll_mean_56",
    "roll_std_56",
    "roll_min_56",
    "roll_max_56",
    "roll_mean_180",
    "roll_std_180",
    "roll_min_180",
    "roll_max_180",
]
DEFAULT_FEATURE_COLS = DEFAULT_NUM_FEATURES + CAT_FEATURES

YLAGS_FEATURE_COLS = DEFAULT_NUM_FEATURES + ["lag_91", "lag_182", "lag_364"] + CAT_FEATURES


class MultiHorizonTrainer:
    """Direct multi-horizon LightGBM trainer (28 independent models).

    cfg keys (matching Hydra config YAML):
      objective          : "tweedie" | "regression"
      tvp                : float, Tweedie variance power (only when objective=tweedie)
      learning_rate      : float
      num_leaves         : int
      min_data_in_leaf   : int
      feature_fraction   : float
      bagging_fraction   : float
      lambda_l2          : float
      num_boost_round    : int
      early_stopping_rounds : int
      horizon            : int  (default 28)
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.horizon = int(cfg.get("horizon", HORIZON))
        self.num_boost_round = int(cfg.get("num_boost_round", 3000))
        self.early_stopping_rounds = int(cfg.get("early_stopping_rounds", 75))
        self._lgb_params = self._build_lgb_params(cfg)

    def _build_lgb_params(self, cfg: dict[str, Any]) -> dict[str, Any]:
        objective = cfg["objective"]
        params: dict[str, Any] = {
            "objective": objective,
            "verbose": -1,
            "num_threads": 0,
            "seed": 42,
            "bagging_freq": 1,
            "learning_rate": cfg.get("learning_rate", 0.025),
            "num_leaves": int(cfg.get("num_leaves", 64)),
            "min_data_in_leaf": int(cfg.get("min_data_in_leaf", 100)),
            "feature_fraction": cfg.get("feature_fraction", 0.7),
            "bagging_fraction": cfg.get("bagging_fraction", 0.9),
            "lambda_l2": cfg.get("lambda_l2", 0.1),
        }
        if objective == "tweedie":
            params["metric"] = "tweedie"
            params["tweedie_variance_power"] = float(cfg.get("tvp", 1.3))
        else:
            params["metric"] = cfg.get("metric", "rmse")
        return params

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        features_dir: str,
        model_dir: str,
        feature_cols: list[str] | None = None,
        raw_dir: str = "data/raw/m5-forecasting-accuracy",
        num_boost_round_override: int | None = None,
        horizon_override: int | None = None,
    ) -> dict[str, Any]:
        """Train horizon_override (or 28) direct models.

        Args:
            features_dir:  path to feature parquet directory
            model_dir:     where to write h_{h:02d}.pkl files
            feature_cols:  columns to use as features; defaults to DEFAULT_FEATURE_COLS
            raw_dir:       path to raw M5 CSVs (for WRMSSE computation)
            num_boost_round_override:  cap num_boost_round (useful in test_mode)
            horizon_override:  train only first N horizons (useful in test_mode)

        Returns:
            {"model_dir", "val_wrmsse", "n_series", "horizon_scores"}
        """
        from shelfsense.evaluation.wrmsse import compute_wrmsse

        if feature_cols is None:
            feature_cols = DEFAULT_FEATURE_COLS
        cat_features = [c for c in CAT_FEATURES if c in feature_cols]
        n_boost = num_boost_round_override or self.num_boost_round
        n_horizon = horizon_override or self.horizon

        os.makedirs(model_dir, exist_ok=True)

        # ── load training data ──────────────────────────────────────────
        load_cols = list(dict.fromkeys(["id"] + feature_cols + ["d_num", "sales"]))
        df = pd.read_parquet(
            features_dir,
            filters=[("d_num", ">=", FEAT_START), ("d_num", "<=", LAST_TRAIN)],
            columns=load_cols,
        )
        for col, dtype in CAT_DTYPES.items():
            if col in df.columns:
                df[col] = df[col].astype(dtype)

        # drop rows where any lag feature is NaN (earliest rows lack lag history)
        lag_cols_present = [c for c in feature_cols if c.startswith("lag_")]
        if lag_cols_present:
            df = df.dropna(subset=lag_cols_present[:1]).reset_index(drop=True)

        # ── load val origin (d_LAST_TRAIN per series) ───────────────────
        origin_cols = list(dict.fromkeys(["id"] + feature_cols + ["d_num"]))
        df_origin = pd.read_parquet(
            features_dir,
            filters=[("d_num", "==", LAST_TRAIN)],
            columns=origin_cols,
        )
        for col, dtype in CAT_DTYPES.items():
            if col in df_origin.columns:
                df_origin[col] = df_origin[col].astype(dtype)
        df_origin = df_origin.sort_values("id").reset_index(drop=True)
        series_ids = df_origin["id"].values
        n_series = len(series_ids)

        # ── train n_horizon models ──────────────────────────────────────
        models: dict[int, Any] = {}
        h_scores: dict[int, dict] = {}

        for h in range(1, n_horizon + 1):
            path = os.path.join(model_dir, f"h_{h:02d}.pkl")
            if os.path.exists(path):
                with open(path, "rb") as fh:
                    models[h] = pickle.load(fh)
                continue

            y_h = df.groupby("id")["sales"].shift(-h)
            valid_h = y_h.notna()

            train_mask = (df["d_num"] >= FEAT_START) & (df["d_num"] <= VAL_START - h - 1) & valid_h
            val_mask = (df["d_num"] >= VAL_START - h) & (df["d_num"] <= LAST_TRAIN - h) & valid_h

            X_tr = df.loc[train_mask, feature_cols]
            y_tr = y_h[train_mask].astype(np.float32)
            X_vl = df.loc[val_mask, feature_cols]
            y_vl = y_h[val_mask].astype(np.float32)

            ds_tr = lgb.Dataset(
                X_tr,
                label=y_tr.values,
                categorical_feature=cat_features,
                free_raw_data=False,
            )
            ds_vl = lgb.Dataset(
                X_vl,
                label=y_vl.values,
                categorical_feature=cat_features,
                reference=ds_tr,
                free_raw_data=False,
            )

            model = lgb.train(
                self._lgb_params,
                ds_tr,
                num_boost_round=n_boost,
                valid_sets=[ds_vl],
                callbacks=[
                    lgb.early_stopping(self.early_stopping_rounds, verbose=False),
                    lgb.log_evaluation(500),
                ],
            )

            with open(path, "wb") as fh:
                pickle.dump(model, fh)

            metric_key = next(iter(model.best_score["valid_0"]))
            h_scores[h] = {
                "best_iter": model.best_iteration,
                "val_metric": float(model.best_score["valid_0"][metric_key]),
            }
            models[h] = model

            del ds_tr, ds_vl, X_tr, y_tr, X_vl, y_vl, y_h
            gc.collect()

        # ── val WRMSSE ──────────────────────────────────────────────────
        val_preds = np.zeros((n_series, n_horizon), dtype=np.float32)
        for h in range(1, n_horizon + 1):
            val_preds[:, h - 1] = np.clip(models[h].predict(df_origin[feature_cols]), 0.0, None)

        sales_eval = pd.read_csv(os.path.join(raw_dir, "sales_train_evaluation.csv"))
        prices_df = pd.read_csv(os.path.join(raw_dir, "sell_prices.csv"))
        calendar_df = pd.read_csv(os.path.join(raw_dir, "calendar.csv"))

        actual_cols = [f"d_{LAST_TRAIN + h}" for h in range(1, n_horizon + 1)]
        sub = (
            sales_eval[sales_eval["id"].isin(series_ids)]
            .set_index("id")
            .reindex(series_ids)
            .reset_index()
        )
        actuals = sub[actual_cols].values.astype(np.float32)

        if n_horizon == HORIZON:
            val_wrmsse, _ = compute_wrmsse(
                preds=val_preds,
                actuals=actuals,
                sales_df=sub,
                prices_df=prices_df,
                calendar_df=calendar_df,
                last_train_day=LAST_TRAIN,
            )
        else:
            # Partial horizon — pad with zeros to keep compute_wrmsse happy
            preds_full = np.zeros((n_series, HORIZON), dtype=np.float32)
            actuals_full = np.zeros((n_series, HORIZON), dtype=np.float32)
            preds_full[:, :n_horizon] = val_preds
            actuals_full[:, :n_horizon] = actuals
            actual_cols_full = [f"d_{LAST_TRAIN + h}" for h in range(1, HORIZON + 1)]
            sub_full = (
                sales_eval[sales_eval["id"].isin(series_ids)]
                .set_index("id")
                .reindex(series_ids)
                .reset_index()
            )
            # fill missing day cols with zeros
            for c in actual_cols_full:
                if c not in sub_full.columns:
                    sub_full[c] = 0.0
            actuals_full = sub_full[actual_cols_full].values.astype(np.float32)
            val_wrmsse, _ = compute_wrmsse(
                preds=preds_full,
                actuals=actuals_full,
                sales_df=sub_full,
                prices_df=prices_df,
                calendar_df=calendar_df,
                last_train_day=LAST_TRAIN,
            )

        return {
            "model_dir": model_dir,
            "val_wrmsse": float(val_wrmsse),
            "n_series": n_series,
            "horizon_scores": h_scores,
        }

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict(
        self,
        model_dir: str,
        features_dir: str,
        forecast_origin_day: int,
        feature_cols: list[str] | None = None,
        horizon_override: int | None = None,
    ) -> pd.DataFrame:
        """Load saved models and predict from forecast_origin_day.

        Returns DataFrame with columns ["id", "F1", ..., "F{horizon}"].
        """
        if feature_cols is None:
            feature_cols = DEFAULT_FEATURE_COLS
        n_horizon = horizon_override or self.horizon

        origin_cols = list(dict.fromkeys(["id"] + feature_cols + ["d_num"]))
        df_origin = pd.read_parquet(
            features_dir,
            filters=[("d_num", "==", forecast_origin_day)],
            columns=origin_cols,
        )
        for col, dtype in CAT_DTYPES.items():
            if col in df_origin.columns:
                df_origin[col] = df_origin[col].astype(dtype)
        df_origin = df_origin.sort_values("id").reset_index(drop=True)

        preds = np.zeros((len(df_origin), n_horizon), dtype=np.float32)
        for h in range(1, n_horizon + 1):
            model_path = os.path.join(model_dir, f"h_{h:02d}.pkl")
            with open(model_path, "rb") as fh:
                model = pickle.load(fh)
            preds[:, h - 1] = np.clip(model.predict(df_origin[feature_cols]), 0.0, None)

        result = pd.DataFrame(preds, columns=[f"F{i}" for i in range(1, n_horizon + 1)])
        result.insert(0, "id", df_origin["id"].values)
        return result
