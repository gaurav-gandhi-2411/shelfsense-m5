"""
WS2.5 Variant — RMSE Multi-Horizon

Train 28 direct multi-horizon LightGBM models with objective="regression" (RMSE)
instead of Tweedie. All other settings match Day 9 exactly so the comparison is
apples-to-apples: same features, same origin (d_1913), same early-stopping budget.

Day 9 best params are hardcoded (no Optuna re-tune) to isolate the effect of the
objective function. Optimal hyperparams for RMSE may differ, but re-tuning would
confound the comparison.

After training:
  a. Same-origin val WRMSSE vs tvp=1.3 baseline (0.6860)
  b. 50-trial Optuna ensemble {tvp=1.3, RMSE-MH} on same-origin val
  c. Build ensemble submission if val improvement > 0.005

Usage:
  python scripts/14_lgbm_rmse_mh.py
"""
from __future__ import annotations
import sys, os, time, json, pickle, gc, warnings
warnings.filterwarnings("ignore")

_SP = "C:/Users/gaura/anaconda3/Lib/site-packages"
if _SP not in sys.path:
    sys.path.insert(0, _SP)

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_ROOT)
sys.path.insert(0, os.path.join(PROJ_ROOT, "src"))

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from features.hierarchy import CAT_DTYPES
from evaluation.wrmsse import compute_wrmsse

print("=" * 65)
print("WS2.5 — RMSE Multi-Horizon (28 models)")
print("=" * 65)

# ── paths ─────────────────────────────────────────────────────────────────────
DATA_RAW  = os.path.join(PROJ_ROOT, "data", "raw", "m5-forecasting-accuracy")
FEAT_DIR  = os.path.join(PROJ_ROOT, "data", "processed", "features")
PREDS_DIR = os.path.join(PROJ_ROOT, "data", "predictions")
REPORTS   = os.path.join(PROJ_ROOT, "reports")
SUBS      = os.path.join(PROJ_ROOT, "submissions")
MODELS    = os.path.join(PROJ_ROOT, "data", "models")
RMSE_DIR  = os.path.join(MODELS, "rmse_mh")
TVP13_DIR = os.path.join(MODELS, "tvp_1p3")
os.makedirs(RMSE_DIR, exist_ok=True)

LAST_TRAIN   = 1913
VAL_START    = 1886
FEAT_START   = 1000
HORIZON      = 28

# Day 9 best params — same as tvp=1.3 script, objective changed to RMSE
BEST_PARAMS = {
    "objective":        "regression",
    "metric":           "rmse",
    "verbose":         -1,
    "num_threads":      0,
    "seed":             42,
    "bagging_freq":     1,
    "learning_rate":    0.025,
    "num_leaves":       64,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.9,
    "lambda_l2":        0.1,
}

CAT_FEATURES = ["cat_id", "dept_id", "store_id", "state_id"]
NUM_FEATURES = [
    "weekday", "month", "quarter", "year", "day_of_month", "week_of_year",
    "is_weekend", "is_holiday", "is_snap_ca", "is_snap_tx", "is_snap_wi",
    "days_since_event", "days_until_next_event",
    "sell_price", "price_change_pct", "price_relative_mean",
    "price_volatility", "has_price_change",
    "lag_7", "lag_14", "lag_28", "lag_56",
    "roll_mean_7",  "roll_std_7",  "roll_min_7",  "roll_max_7",
    "roll_mean_28", "roll_std_28", "roll_min_28", "roll_max_28",
    "roll_mean_56", "roll_std_56", "roll_min_56", "roll_max_56",
    "roll_mean_180","roll_std_180","roll_min_180","roll_max_180",
]
ALL_FEATURES = NUM_FEATURES + CAT_FEATURES
f_cols = [f"F{i}" for i in range(1, HORIZON + 1)]
actual_val_cols = [f"d_{LAST_TRAIN + h}" for h in range(1, HORIZON + 1)]

print(f"\nDay 9 params (objective overridden to RMSE):")
for k, v in BEST_PARAMS.items():
    print(f"  {k}={v}")

# ── 1. Load training data ──────────────────────────────────────────────────────
print(f"\n[1] Loading training data (d_num {FEAT_START}-{LAST_TRAIN})...")
t0 = time.time()
df = pd.read_parquet(
    FEAT_DIR,
    filters=[("d_num", ">=", FEAT_START), ("d_num", "<=", LAST_TRAIN)],
    columns=["id", "cat_id", "dept_id", "store_id", "state_id",
             "d_num", "sales"] + NUM_FEATURES,
)
for col, dtype in CAT_DTYPES.items():
    if col in df.columns:
        df[col] = df[col].astype(dtype)
df = df.dropna(subset=["lag_7","lag_14","lag_28","lag_56"]).reset_index(drop=True)
print(f"    Rows: {len(df):,}  ({time.time()-t0:.1f}s)")

# ── 1b. Load raw CSVs for scoring ─────────────────────────────────────────────
print("\n[1b] Loading raw CSVs...")
t0 = time.time()
sales_eval  = pd.read_csv(os.path.join(DATA_RAW, "sales_train_evaluation.csv"))
prices_df   = pd.read_csv(os.path.join(DATA_RAW, "sell_prices.csv"))
calendar_df = pd.read_csv(os.path.join(DATA_RAW, "calendar.csv"))
print(f"     Loaded in {time.time()-t0:.1f}s")

# ── 2. Load inference origins ─────────────────────────────────────────────────
print(f"\n[2] Loading inference origins (d_{LAST_TRAIN} and d_{LAST_TRAIN + HORIZON})...")
df_origins = pd.read_parquet(
    FEAT_DIR,
    filters=[("d_num", ">=", LAST_TRAIN), ("d_num", "<=", LAST_TRAIN + HORIZON)],
    columns=["id", "cat_id", "dept_id", "store_id", "state_id", "d_num"] + NUM_FEATURES,
)
for col, dtype in CAT_DTYPES.items():
    if col in df_origins.columns:
        df_origins[col] = df_origins[col].astype(dtype)

df_val_origin  = df_origins[df_origins["d_num"] == LAST_TRAIN].sort_values("id").reset_index(drop=True)
df_eval_origin = df_origins[df_origins["d_num"] == LAST_TRAIN + HORIZON].sort_values("id").reset_index(drop=True)
series_ids = df_val_origin["id"].values
n_series   = len(series_ids)
print(f"    Series: {n_series:,}")
del df_origins

def score_preds(preds_mat, sids):
    sub  = sales_eval[sales_eval["id"].isin(sids)].set_index("id").reindex(sids).reset_index()
    acts = sub[actual_val_cols].values.astype(np.float32)
    s, _ = compute_wrmsse(
        preds=preds_mat, actuals=acts,
        sales_df=sub, prices_df=prices_df,
        calendar_df=calendar_df, last_train_day=LAST_TRAIN,
    )
    return float(s)

# ── 3. Train 28 RMSE models ────────────────────────────────────────────────────
print(f"\n[3] Training {HORIZON} RMSE models (h=1..{HORIZON})...")
models = {}
t_total = time.time()

for h in range(1, HORIZON + 1):
    path = os.path.join(RMSE_DIR, f"h_{h:02d}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            models[h] = pickle.load(f)
        print(f"  h={h:2d}: loaded from checkpoint")
        continue

    y_h     = df.groupby("id")["sales"].shift(-h)
    valid_h = y_h.notna()

    train_mask = (
        (df["d_num"] >= FEAT_START) &
        (df["d_num"] <= VAL_START - h - 1) & valid_h
    )
    val_mask = (
        (df["d_num"] >= VAL_START - h) &
        (df["d_num"] <= LAST_TRAIN - h) & valid_h
    )

    X_tr = df.loc[train_mask, ALL_FEATURES]
    y_tr = y_h[train_mask].astype(np.float32)
    X_vl = df.loc[val_mask, ALL_FEATURES]
    y_vl = y_h[val_mask].astype(np.float32)

    ds_tr = lgb.Dataset(X_tr, label=y_tr.values, categorical_feature=CAT_FEATURES, free_raw_data=False)
    ds_vl = lgb.Dataset(X_vl, label=y_vl.values, categorical_feature=CAT_FEATURES, reference=ds_tr, free_raw_data=False)

    t0 = time.time()
    model = lgb.train(
        BEST_PARAMS, ds_tr,
        num_boost_round=3000,
        valid_sets=[ds_vl],
        callbacks=[lgb.early_stopping(75, verbose=False), lgb.log_evaluation(500)],
    )
    elapsed = time.time() - t0

    with open(path, "wb") as f:
        pickle.dump(model, f)

    val_metric = model.best_score["valid_0"]["rmse"]
    print(f"  h={h:2d}: iter={model.best_iteration:4d}  val_rmse={val_metric:.4f}  {elapsed:.0f}s  saved")
    models[h] = model

    del ds_tr, ds_vl, X_tr, y_tr, X_vl, y_vl, y_h
    gc.collect()

print(f"\n    All {HORIZON} models in {(time.time()-t_total)/60:.1f} min")

# ── 4. Val WRMSSE (same-origin d_1913) ────────────────────────────────────────
print(f"\n[4] Val WRMSSE (same-origin d_{LAST_TRAIN})...")
val_rmse = np.zeros((n_series, HORIZON), dtype=np.float32)
t0 = time.time()
for h in range(1, HORIZON + 1):
    val_rmse[:, h-1] = np.clip(models[h].predict(df_val_origin[ALL_FEATURES]), 0.0, None).astype(np.float32)
print(f"    Inference done in {time.time()-t0:.1f}s")

wrmsse_rmse = score_preds(val_rmse, series_ids)
print(f"\n    RMSE-MH val WRMSSE (origin d_{LAST_TRAIN}): {wrmsse_rmse:.4f}")
print(f"    tvp=1.3 reference:                          0.6860")
print(f"    Delta:                                      {wrmsse_rmse - 0.6860:+.4f}")

# ── 5. Eval predictions ────────────────────────────────────────────────────────
print(f"\n[5] Eval predictions (from d_{LAST_TRAIN + HORIZON})...")
eval_rmse = np.zeros((n_series, HORIZON), dtype=np.float32)
for h in range(1, HORIZON + 1):
    eval_rmse[:, h-1] = np.clip(models[h].predict(df_eval_origin[ALL_FEATURES]), 0.0, None).astype(np.float32)

# ── 6. Save parquets ──────────────────────────────────────────────────────────
print("\n[6] Saving prediction parquets...")
pd.DataFrame({"id": series_ids, **{f"F{h}": val_rmse[:, h-1] for h in range(1, HORIZON+1)}}).to_parquet(
    os.path.join(PREDS_DIR, "lgbm_rmse_mh_val.parquet"), index=False
)
pd.DataFrame({"id": series_ids, **{f"F{h}": eval_rmse[:, h-1] for h in range(1, HORIZON+1)}}).to_parquet(
    os.path.join(PREDS_DIR, "lgbm_rmse_mh_eval.parquet"), index=False
)
print("    Saved lgbm_rmse_mh_val.parquet and lgbm_rmse_mh_eval.parquet")

# ── 7. Optuna ensemble: {tvp=1.3, RMSE-MH} same-origin val ───────────────────
print("\n[7] Optuna ensemble {tvp=1.3, RMSE-MH} — 50 trials, same-origin val WRMSSE...")

val_1p3 = pd.read_parquet(os.path.join(PREDS_DIR, "lgbm_tvp_1p3_val.parquet"))
val_1p3 = val_1p3.set_index("id").loc[series_ids, f_cols].values.astype(np.float32)

eval_1p3 = pd.read_parquet(os.path.join(PREDS_DIR, "lgbm_tvp_1p3_eval.parquet"))
eval_1p3 = eval_1p3.set_index("id").loc[series_ids, f_cols].values.astype(np.float32)

def optuna_objective(trial):
    w1 = trial.suggest_float("w_rmse", 0.0, 1.0)
    w2 = trial.suggest_float("w_13",   0.0, 1.0)
    total = w1 + w2
    if total < 1e-6:
        return 9999.0
    blend = (w1 * val_rmse + w2 * val_1p3) / total
    return score_preds(blend, series_ids)

study = optuna.create_study(direction="minimize")
t0 = time.time()
study.optimize(optuna_objective, n_trials=50, show_progress_bar=False)
opt_time = time.time() - t0

bp = study.best_params
total_w = bp["w_rmse"] + bp["w_13"]
w_rmse_n = bp["w_rmse"] / total_w
w_13_n   = bp["w_13"]  / total_w

print(f"    Optuna done in {opt_time:.0f}s  best_val={study.best_value:.4f}")
print(f"    Optimal weights: RMSE-MH={w_rmse_n:.3f}  tvp=1.3={w_13_n:.3f}")
print(f"    tvp=1.3 standalone:   0.6860")
print(f"    Improvement:          {0.6860 - study.best_value:+.4f}  (positive = better)")

# ── 8. Build submissions ───────────────────────────────────────────────────────
sn28 = pd.read_csv(os.path.join(SUBS, "seasonal_naive_28_submission.csv"))
mh_sub = pd.read_csv(os.path.join(SUBS, "mh_blend.csv")).set_index("id")
val_ids = [s.replace("_evaluation", "_validation") for s in series_ids]
val_oracle_mat = mh_sub.loc[val_ids, f_cols].values.astype(np.float32)

def build_submission(eval_mat, fname):
    base = sn28.copy().set_index("id")
    vdf  = pd.DataFrame(val_oracle_mat, columns=f_cols)
    vdf.insert(0, "id", val_ids)
    val_idx  = pd.Index(val_ids).intersection(base.index)
    base.loc[val_idx] = vdf.set_index("id").loc[val_idx]
    edf  = pd.DataFrame(eval_mat, columns=f_cols)
    edf.insert(0, "id", list(series_ids))
    eval_idx = pd.Index(series_ids).intersection(base.index)
    base.loc[eval_idx] = edf.set_index("id").loc[eval_idx]
    path = os.path.join(SUBS, fname)
    base.reset_index().to_csv(path, index=False)
    print(f"    Saved {fname}  ({len(val_idx)} val + {len(eval_idx)} eval rows)")

print("\n[8] Building submissions...")
build_submission(eval_rmse, "lgbm_rmse_mh.csv")

IMPROVEMENT_THRESHOLD = 0.005
improvement = 0.6860 - study.best_value
if improvement > IMPROVEMENT_THRESHOLD:
    print(f"    Ensemble val improvement {improvement:.4f} > {IMPROVEMENT_THRESHOLD} threshold — building ensemble CSV...")
    ens_eval = np.clip(w_rmse_n * eval_rmse + w_13_n * eval_1p3, 0.0, None).astype(np.float32)
    build_submission(ens_eval, "ens_rmse_tvp13_optuna.csv")
    print(f"    Built ens_rmse_tvp13_optuna.csv — READY TO SUBMIT")
else:
    print(f"    Ensemble val improvement {improvement:.4f} <= {IMPROVEMENT_THRESHOLD} threshold — skipping ensemble CSV")

# ── 9. Save results JSON ──────────────────────────────────────────────────────
def make_serial(obj):
    if isinstance(obj, (np.floating, float)): return float(obj)
    if isinstance(obj, (np.integer, int)):    return int(obj)
    if isinstance(obj, dict):  return {k: make_serial(v) for k, v in obj.items()}
    if isinstance(obj, list):  return [make_serial(x) for x in obj]
    return obj

results = {
    "rmse_mh_val_wrmsse": wrmsse_rmse,
    "tvp13_val_wrmsse":   0.6860,
    "delta":              wrmsse_rmse - 0.6860,
    "optuna_ensemble": {
        "n_trials":    50,
        "best_val":    study.best_value,
        "improvement_vs_tvp13": improvement,
        "weights":     {"w_rmse": w_rmse_n, "w_13": w_13_n},
        "threshold":   IMPROVEMENT_THRESHOLD,
        "built_ensemble_csv": improvement > IMPROVEMENT_THRESHOLD,
    },
}

json_path = os.path.join(REPORTS, "day14_rmse_mh_scores.json")
with open(json_path, "w") as f:
    json.dump(make_serial(results), f, indent=2)
print(f"\n    Saved {json_path}")

# ── summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("RMSE-MH SUMMARY")
print("=" * 65)
print(f"\n  (a) RMSE-MH val WRMSSE (same-origin d_{LAST_TRAIN}): {wrmsse_rmse:.4f}")
print(f"      tvp=1.3 reference:                              0.6860")
print(f"      Delta:                                          {wrmsse_rmse - 0.6860:+.4f}")
print(f"\n  (b) Optuna ensemble {{tvp=1.3, RMSE-MH}}:")
print(f"      Best val WRMSSE: {study.best_value:.4f}")
print(f"      Weights: RMSE-MH={w_rmse_n:.3f}  tvp=1.3={w_13_n:.3f}")
print(f"      Improvement vs tvp=1.3: {improvement:+.4f}")
print(f"\n  (c) Ensemble CSV built: {improvement > IMPROVEMENT_THRESHOLD}")
if improvement > IMPROVEMENT_THRESHOLD:
    print(f"      File: submissions/ens_rmse_tvp13_optuna.csv")
    print(f"      Submit:")
    print(f"        kaggle competitions submit -c m5-forecasting-accuracy \\")
    print(f"          -f submissions/ens_rmse_tvp13_optuna.csv \\")
    print(f"          -m \"WS2.5 RMSE+tvp13 Optuna ensemble\"")
    print(f"      Also submit standalone RMSE-MH for reference:")
    print(f"        kaggle competitions submit -c m5-forecasting-accuracy \\")
    print(f"          -f submissions/lgbm_rmse_mh.csv \\")
    print(f"          -m \"WS2.5 RMSE multi-horizon standalone\"")
else:
    print(f"      Standalone RMSE-MH CSV available but ensemble not justified:")
    print(f"        submissions/lgbm_rmse_mh.csv")
print("\nDone!")
