"""Store × department LightGBM trainer with recursive 28-step forecasting.

Trains one single-horizon Tweedie model per (store, dept) slice (70 total).
Predictions use recursive autoregression via predict_horizon() from recursive.py.
Models are cached by a design-hash so any constant change forces full retrain.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import pickle
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from shelfsense.features.hierarchy import CAT_DTYPES

# Feature columns must match what predict_horizon() builds internally
from shelfsense.models.lightgbm.multihorizon import DEFAULT_FEATURE_COLS

LAST_TRAIN = 1913
VAL_START = 1886
FEAT_START = 1000
HORIZON = 28

STORES = ["CA_1", "CA_2", "CA_3", "CA_4", "TX_1", "TX_2", "TX_3", "WI_1", "WI_2", "WI_3"]
DEPTS = ["FOODS_1", "FOODS_2", "FOODS_3", "HOBBIES_1", "HOBBIES_2", "HOUSEHOLD_1", "HOUSEHOLD_2"]
CAT_FEATURES = ["cat_id", "dept_id", "store_id", "state_id"]


def _model_filename(store: str, dept: str, design_hash: str) -> str:
    return f"lgbm_SD_{store}_{dept}_p{design_hash}.pkl"


def _compute_design_hash(cfg: dict[str, Any]) -> str:
    design = {
        "objective": cfg.get("objective", "tweedie"),
        "tvp": cfg.get("tweedie_variance_power", cfg.get("tvp", 1.3)),
        "feat_start": FEAT_START,
        "last_train": LAST_TRAIN,
        "val_start": VAL_START,
        "n_optuna": cfg.get("optuna_trials", 10),
        "feature_set": "v1_38num_4cat",
    }
    return hashlib.md5(json.dumps(design, sort_keys=True).encode()).hexdigest()[:8]


def _build_hist_from_wide(
    sales_df: pd.DataFrame,
    series_ids: list[str],
    last_day: int,
    history_days: int = 200,
) -> pd.DataFrame:
    """Build long-format history DataFrame from wide-format sales CSV."""
    first_day = last_day - history_days + 1
    day_cols = [f"d_{d}" for d in range(first_day, last_day + 1) if f"d_{d}" in sales_df.columns]
    meta = ["id", "item_id", "cat_id", "dept_id", "store_id", "state_id"]
    sub = (
        sales_df[sales_df["id"].isin(series_ids)].set_index("id").reindex(series_ids).reset_index()
    )
    df = sub[meta + day_cols].melt(id_vars=meta, var_name="d", value_name="sales")
    df["d_num"] = df["d"].str.replace("d_", "", regex=False).astype(np.int32)
    df["sales"] = df["sales"].fillna(0.0).astype(np.float32)
    return df.drop(columns=["d"])


class StoreDeptTrainer:
    """Per-slice LightGBM trainer for store × department forecasting.

    cfg keys (matching store_dept.yaml):
      objective            : "tweedie"
      tweedie_variance_power / tvp : float
      optuna_trials        : int  (trials per slice)
      lr_min/lr_max        : float (log-scale Optuna search)
      num_leaves_min/max   : int
      min_data_in_leaf_min/max : int
      feature_fraction_min/max : float
      bagging_fraction_min/max : float
      num_boost_round      : int
      early_stopping_rounds : int
      stores               : list[str]
      departments          : list[str]
      history_days         : int
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.n_optuna = int(cfg.get("optuna_trials", 10))
        self.stores = list(cfg.get("stores", STORES))
        self.depts = list(cfg.get("departments", DEPTS))
        self.num_boost_round = int(cfg.get("num_boost_round", 3000))
        self.early_stopping_rounds = int(cfg.get("early_stopping_rounds", 75))
        self.history_days = int(cfg.get("history_days", 200))
        self.design_hash = _compute_design_hash(cfg)

        tvp = cfg.get("tweedie_variance_power", cfg.get("tvp", 1.3))
        self._base_params: dict[str, Any] = {
            "objective": cfg.get("objective", "tweedie"),
            "metric": "tweedie",
            "verbose": -1,
            "num_threads": int(cfg.get("num_threads", 0)),
            "bagging_freq": 1,
            "tweedie_variance_power": float(tvp),
            "seed": int(cfg.get("seed", 42)),
        }
        self._search_space = {
            "lr_min": float(cfg.get("lr_min", 0.01)),
            "lr_max": float(cfg.get("lr_max", 0.1)),
            "num_leaves_min": int(cfg.get("num_leaves_min", 31)),
            "num_leaves_max": int(cfg.get("num_leaves_max", 127)),
            "min_leaf_min": int(cfg.get("min_data_in_leaf_min", 20)),
            "min_leaf_max": int(cfg.get("min_data_in_leaf_max", 100)),
            "ff_min": float(cfg.get("feature_fraction_min", 0.5)),
            "ff_max": float(cfg.get("feature_fraction_max", 1.0)),
            "bf_min": float(cfg.get("bagging_fraction_min", 0.5)),
            "bf_max": float(cfg.get("bagging_fraction_max", 1.0)),
        }

    # ------------------------------------------------------------------
    # Internal: Optuna sweep for one slice
    # ------------------------------------------------------------------

    def _train_slice(
        self,
        df_tr: pd.DataFrame,
        df_vl: pd.DataFrame,
        feature_cols: list[str],
        n_trials: int,
        num_boost_round: int,
    ) -> tuple[Any, float, dict, int]:
        """Run Optuna on one (store, dept) slice. Returns (model, val_tweedie, params, iter)."""
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        cat_features = [c for c in CAT_FEATURES if c in feature_cols]
        X_tr = df_tr[feature_cols]
        y_tr = df_tr["sales"].values.astype(np.float32)
        X_vl = df_vl[feature_cols]
        y_vl = df_vl["sales"].values.astype(np.float32)

        ds_tr = lgb.Dataset(
            X_tr,
            label=y_tr,
            categorical_feature=cat_features,
            free_raw_data=False,
        )
        ds_vl = lgb.Dataset(
            X_vl,
            label=y_vl,
            categorical_feature=cat_features,
            reference=ds_tr,
            free_raw_data=False,
        )

        ss = self._search_space
        best: dict[str, Any] = {"val": float("inf"), "model": None, "params": {}, "iter": 0}

        def _obj(trial: "optuna.Trial") -> float:
            params = {
                **self._base_params,
                "learning_rate": trial.suggest_float("lr", ss["lr_min"], ss["lr_max"], log=True),
                "num_leaves": trial.suggest_int(
                    "num_leaves", ss["num_leaves_min"], ss["num_leaves_max"]
                ),
                "min_data_in_leaf": trial.suggest_int(
                    "min_leaf", ss["min_leaf_min"], ss["min_leaf_max"]
                ),
                "feature_fraction": trial.suggest_float("ff", ss["ff_min"], ss["ff_max"]),
                "bagging_fraction": trial.suggest_float("bf", ss["bf_min"], ss["bf_max"]),
            }
            model = lgb.train(
                params,
                ds_tr,
                num_boost_round=num_boost_round,
                valid_sets=[ds_vl],
                callbacks=[
                    lgb.early_stopping(self.early_stopping_rounds, verbose=False),
                    lgb.log_evaluation(-1),
                ],
            )
            val = float(model.best_score["valid_0"]["tweedie"])
            if val < best["val"]:
                best.update(
                    val=val, model=model, params=dict(trial.params), iter=model.best_iteration
                )
            return val

        study = optuna.create_study(direction="minimize")
        study.optimize(_obj, n_trials=n_trials, show_progress_bar=False)
        return best["model"], best["val"], best["params"], best["iter"]

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
        optuna_trials_override: int | None = None,
        slices_override: list[tuple[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Train one model per (store, dept) slice.

        Args:
            features_dir:  path to feature parquet directory
            model_dir:     where to write lgbm_SD_*.pkl files
            feature_cols:  feature columns; defaults to DEFAULT_FEATURE_COLS
            raw_dir:       M5 CSV directory (for WRMSSE + recursive prediction)
            num_boost_round_override:  cap num_boost_round (useful in test_mode)
            optuna_trials_override:  cap Optuna trials per slice (useful in test_mode)
            slices_override:  train only these (store, dept) pairs (useful in test_mode)

        Returns:
            {"model_dir", "val_wrmsse", "n_slices", "n_slices_cached", "slice_results"}
        """
        from shelfsense.evaluation.wrmsse import compute_wrmsse
        from shelfsense.models.lightgbm.recursive import predict_horizon

        if feature_cols is None:
            feature_cols = DEFAULT_FEATURE_COLS
        n_boost = num_boost_round_override or self.num_boost_round
        n_optuna = optuna_trials_override or self.n_optuna
        slices = slices_override or [(s, d) for s in self.stores for d in self.depts]

        os.makedirs(model_dir, exist_ok=True)

        # Load raw CSVs once (needed for history building + WRMSSE)
        sales_eval = pd.read_csv(os.path.join(raw_dir, "sales_train_evaluation.csv"))
        prices_df = pd.read_csv(os.path.join(raw_dir, "sell_prices.csv"))
        calendar_df = pd.read_csv(os.path.join(raw_dir, "calendar.csv"))

        actual_val_cols = [f"d_{LAST_TRAIN + h}" for h in range(1, HORIZON + 1)]

        load_cols = list(
            dict.fromkeys(["id", "item_id", "dept_id", "d_num", "sales"] + feature_cols)
        )

        slice_results: dict[tuple, dict] = {}
        n_trained = 0
        n_cached = 0
        store_cache: dict[str, pd.DataFrame] = {}

        for store, dept in slices:
            pkl_path = os.path.join(model_dir, _model_filename(store, dept, self.design_hash))

            # ── cache hit ──────────────────────────────────────────────
            if os.path.exists(pkl_path) and num_boost_round_override is None:
                with open(pkl_path, "rb") as fh:
                    cached = pickle.load(fh)
                slice_results[(store, dept)] = {k: v for k, v in cached.items() if k != "model"}
                n_cached += 1
                continue

            # ── load store parquet (reuse across depts in same store) ──
            if store not in store_cache:
                parquet_path = os.path.join(features_dir, f"store_{store}.parquet")
                df_store = pd.read_parquet(
                    parquet_path,
                    columns=load_cols,
                    filters=[("d_num", ">=", FEAT_START), ("d_num", "<=", LAST_TRAIN)],
                )
                for col, dtype in CAT_DTYPES.items():
                    if col in df_store.columns:
                        df_store[col] = df_store[col].astype(dtype)
                df_store = df_store.dropna(
                    subset=["lag_7", "lag_14", "lag_28", "lag_56"]
                ).reset_index(drop=True)
                store_cache[store] = df_store

            df = store_cache[store]
            df_dept = df[df["dept_id"] == dept]
            if len(df_dept) == 0:
                continue

            series_ids = sorted(df_dept["id"].unique())
            n_series = len(series_ids)

            df_tr = df_dept[df_dept["d_num"] <= VAL_START - 1]
            df_vl = df_dept[df_dept["d_num"] >= VAL_START]

            # ── Optuna sweep ───────────────────────────────────────────
            model, val_tweedie, best_params, best_iter = self._train_slice(
                df_tr, df_vl, feature_cols, n_optuna, n_boost
            )

            # ── Recursive val predictions ──────────────────────────────
            hist_val = _build_hist_from_wide(
                sales_eval,
                series_ids,
                last_day=LAST_TRAIN,
                history_days=self.history_days,
            )
            val_preds, _ = predict_horizon(
                model,
                hist_val,
                calendar_df,
                prices_df,
                days_out=HORIZON,
                verbose=False,
            )

            # ── Per-slice WRMSSE ───────────────────────────────────────
            try:
                sub_sales = (
                    sales_eval[sales_eval["id"].isin(series_ids)]
                    .set_index("id")
                    .reindex(series_ids)
                    .reset_index()
                )
                actuals_sl = sub_sales[actual_val_cols].values.astype(np.float32)
                slice_wrmsse, _ = compute_wrmsse(
                    val_preds,
                    actuals_sl,
                    sub_sales,
                    prices_df,
                    calendar_df,
                    LAST_TRAIN,
                )
                slice_wrmsse = float(slice_wrmsse)
            except Exception:
                slice_wrmsse = float("nan")

            result = {
                "store": store,
                "dept": dept,
                "n_series": n_series,
                "val_tweedie": float(val_tweedie),
                "val_wrmsse": slice_wrmsse,
                "best_iter": int(best_iter),
                "best_params": best_params,
                "val_preds": val_preds.astype(np.float32),
                "series_ids": series_ids,
            }

            with open(pkl_path, "wb") as fh:
                pickle.dump({**result, "model": model}, fh)

            slice_results[(store, dept)] = result
            n_trained += 1
            del df_dept, df_tr, df_vl, model
            gc.collect()

        # ── Full-catalogue val WRMSSE (covered series only) ────────────
        all_preds_list, all_ids_list = [], []
        for r in slice_results.values():
            for i, sid in enumerate(r["series_ids"]):
                all_ids_list.append(sid)
                all_preds_list.append(r["val_preds"][i])

        if all_preds_list:
            preds_mat = np.vstack(all_preds_list).astype(np.float32)
            covered_ids = all_ids_list
            sub_all = (
                sales_eval[sales_eval["id"].isin(covered_ids)]
                .set_index("id")
                .reindex(covered_ids)
                .reset_index()
            )
            actuals_all = sub_all[actual_val_cols].values.astype(np.float32)
            try:
                full_wrmsse, _ = compute_wrmsse(
                    preds_mat,
                    actuals_all,
                    sub_all,
                    prices_df,
                    calendar_df,
                    LAST_TRAIN,
                )
                full_wrmsse = float(full_wrmsse)
            except Exception:
                full_wrmsse = float("nan")
        else:
            full_wrmsse = float("nan")

        return {
            "model_dir": model_dir,
            "val_wrmsse": full_wrmsse,
            "n_slices": n_trained,
            "n_slices_cached": n_cached,
            "slice_results": {
                f"{s}_{d}": {
                    "val_wrmsse": r["val_wrmsse"],
                    "val_tweedie": r["val_tweedie"],
                    "n_series": r["n_series"],
                    "best_iter": r["best_iter"],
                }
                for (s, d), r in slice_results.items()
            },
        }

    # ------------------------------------------------------------------
    # val_preds_from_cache (no CSV loads — reads cached val_preds from pkl)
    # ------------------------------------------------------------------

    def val_preds_from_cache(
        self,
        model_dir: str,
        slices_override: list[tuple[str, str]] | None = None,
    ) -> pd.DataFrame:
        """Return cached val predictions (d_1913 origin) stored inside each pkl.

        No CSV loads required — uses the val_preds array saved by fit().
        Returns DataFrame with columns ["id", "F1", ..., "F28"].
        """
        slices = slices_override or [(s, d) for s in self.stores for d in self.depts]
        rows: list[dict] = []
        for store, dept in slices:
            pkl_path = os.path.join(model_dir, _model_filename(store, dept, self.design_hash))
            if not os.path.exists(pkl_path):
                continue
            with open(pkl_path, "rb") as fh:
                cached = pickle.load(fh)
            series_ids = cached["series_ids"]
            val_preds = cached["val_preds"]  # (n_series, 28), stored by fit()
            for i, sid in enumerate(series_ids):
                row = {"id": sid}
                for h in range(1, HORIZON + 1):
                    row[f"F{h}"] = float(val_preds[i, h - 1])
                rows.append(row)
        if not rows:
            return pd.DataFrame(columns=["id"] + [f"F{h}" for h in range(1, HORIZON + 1)])
        return pd.DataFrame(rows).sort_values("id").reset_index(drop=True)

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict(
        self,
        model_dir: str,
        raw_dir: str,
        forecast_origin_day: int,
        feature_cols: list[str] | None = None,
        slices_override: list[tuple[str, str]] | None = None,
    ) -> pd.DataFrame:
        """Recursive 28-step forecast for all covered (store, dept) slices.

        Returns DataFrame with columns ["id", "F1", ..., "F28"].
        """
        from shelfsense.models.lightgbm.recursive import predict_horizon

        if feature_cols is None:
            feature_cols = DEFAULT_FEATURE_COLS
        slices = slices_override or [(s, d) for s in self.stores for d in self.depts]

        sales_eval = pd.read_csv(os.path.join(raw_dir, "sales_train_evaluation.csv"))
        prices_df = pd.read_csv(os.path.join(raw_dir, "sell_prices.csv"))
        calendar_df = pd.read_csv(os.path.join(raw_dir, "calendar.csv"))

        rows: list[dict] = []

        for store, dept in slices:
            pkl_path = os.path.join(model_dir, _model_filename(store, dept, self.design_hash))
            if not os.path.exists(pkl_path):
                continue
            with open(pkl_path, "rb") as fh:
                cached = pickle.load(fh)
            model = cached["model"]
            series_ids = cached["series_ids"]

            hist_df = _build_hist_from_wide(
                sales_eval,
                series_ids,
                last_day=forecast_origin_day,
                history_days=self.history_days,
            )
            preds, pred_ids = predict_horizon(
                model,
                hist_df,
                calendar_df,
                prices_df,
                days_out=HORIZON,
                verbose=False,
            )
            for i, sid in enumerate(pred_ids):
                row = {"id": sid}
                for h in range(1, HORIZON + 1):
                    row[f"F{h}"] = float(preds[i, h - 1])
                rows.append(row)

            del model
            gc.collect()

        if not rows:
            return pd.DataFrame(columns=["id"] + [f"F{h}" for h in range(1, HORIZON + 1)])
        return pd.DataFrame(rows).sort_values("id").reset_index(drop=True)
