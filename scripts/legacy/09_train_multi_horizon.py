"""
Day 9: 28 multi-horizon LightGBM models — one model per forecast horizon h=1..28.

Each model predicts sales[d+h] using features available at time d (no future leakage).
At inference, all 28 models run on a SINGLE origin point (d_1941), so no recursive
compounding is possible — every prediction uses actual historical sales features.

Key difference from recursive forecasting:
  Recursive d_1969: features include 27 predicted (noisy) lag values
  Multi-horizon d_1969: features from d_1941 — all actual sales, zero compounding

Training split for model_h:
  Train:  d_num in [FEAT_START, VAL_START - h - 1]  (target in training territory)
  Val:    d_num in [VAL_START - h, LAST_TRAIN - h]   (target in val territory d_1886..d_1913)
  Target: df.groupby("id")["sales"].shift(-h)         (sales h days ahead)

Hyperparameters: Optuna 15 trials on h=14 (median horizon), shared across all 28 models.
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

print("=" * 60)
print("DAY 9 -- Multi-Horizon LightGBM (28 models)")
print("=" * 60)

DATA_RAW  = os.path.join(PROJ_ROOT, "data", "raw", "m5-forecasting-accuracy")
FEAT_DIR  = os.path.join(PROJ_ROOT, "data", "processed", "features")
MODELS    = os.path.join(PROJ_ROOT, "data", "models")
MH_DIR    = os.path.join(MODELS, "multi_horizon")
REPORTS   = os.path.join(PROJ_ROOT, "reports")
SUBS      = os.path.join(PROJ_ROOT, "submissions")
os.makedirs(MH_DIR, exist_ok=True)

LAST_TRAIN   = 1913
VAL_START    = 1886      # first day of LightGBM val targets (d_1886..d_1913)
FEAT_START   = 1000
OPTUNA_START = 1600
HORIZON      = 28
EVAL_START   = LAST_TRAIN + HORIZON + 1   # 1942
EVAL_END     = EVAL_START + HORIZON - 1   # 1969
N_TRIALS     = 15
OPTUNA_H     = 14        # tune on median horizon, share params across all h

CAT_FEATURES = ["cat_id", "dept_id", "store_id", "state_id"]
NUM_FEATURES = [
    "weekday", "month", "quarter", "year", "day_of_month", "week_of_year",
    "is_weekend", "is_holiday",
    "is_snap_ca", "is_snap_tx", "is_snap_wi",
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

# ── 1. Load training data and raw files ───────────────────────────────────────
print(f"\n[1] Loading training data (d_num {FEAT_START}-{LAST_TRAIN})...")
t0 = time.time()
df = pd.read_parquet(
    FEAT_DIR,
    filters=[("d_num", ">=", FEAT_START), ("d_num", "<=", LAST_TRAIN)],
    columns=["id", "item_id", "cat_id", "dept_id", "store_id", "state_id",
             "d_num", "sales"] + NUM_FEATURES,
)
for col, dtype in CAT_DTYPES.items():
    if col in df.columns:
        df[col] = df[col].astype(dtype)
df = df.dropna(subset=["lag_7","lag_14","lag_28","lag_56"]).reset_index(drop=True)
print(f"    Rows: {len(df):,}  ({time.time()-t0:.1f}s)")

print("\n    Loading raw CSVs...")
t0 = time.time()
sales_eval  = pd.read_csv(os.path.join(DATA_RAW, "sales_train_evaluation.csv"))
prices_df   = pd.read_csv(os.path.join(DATA_RAW, "sell_prices.csv"))
calendar_df = pd.read_csv(os.path.join(DATA_RAW, "calendar.csv"))
print(f"    Loaded in {time.time()-t0:.1f}s")

# ── 2. Load inference origin rows ─────────────────────────────────────────────
print(f"\n[2] Loading inference origins (d_{LAST_TRAIN} and d_{LAST_TRAIN + HORIZON})...")
df_origins = pd.read_parquet(
    FEAT_DIR,
    filters=[("d_num", ">=", LAST_TRAIN), ("d_num", "<=", LAST_TRAIN + HORIZON)],
    columns=["id", "item_id", "cat_id", "dept_id", "store_id", "state_id",
             "d_num"] + NUM_FEATURES,
)
for col, dtype in CAT_DTYPES.items():
    if col in df_origins.columns:
        df_origins[col] = df_origins[col].astype(dtype)

df_val_origin  = df_origins[df_origins["d_num"] == LAST_TRAIN].sort_values("id").reset_index(drop=True)
df_eval_origin = df_origins[df_origins["d_num"] == LAST_TRAIN + HORIZON].sort_values("id").reset_index(drop=True)
series_ids = df_val_origin["id"].values
n_series   = len(series_ids)
print(f"    Val origin (d_{LAST_TRAIN}): {len(df_val_origin):,} series")
print(f"    Eval origin (d_{LAST_TRAIN + HORIZON}): {len(df_eval_origin):,} series")

# ── helpers ───────────────────────────────────────────────────────────────────
actual_val_cols = [f"d_{LAST_TRAIN + h}" for h in range(1, HORIZON + 1)]

def score_preds(preds_mat, sids):
    sub  = sales_eval[sales_eval["id"].isin(sids)].set_index("id").reindex(sids).reset_index()
    acts = sub[actual_val_cols].values.astype(np.float32)
    s, _ = compute_wrmsse(
        preds=preds_mat, actuals=acts,
        sales_df=sub, prices_df=prices_df,
        calendar_df=calendar_df, last_train_day=LAST_TRAIN,
    )
    return float(s)

# ── 3. Optuna hyperparameter search on h=14 ───────────────────────────────────
print(f"\n[3] Optuna ({N_TRIALS} trials) on h={OPTUNA_H}...")

h_opt = OPTUNA_H
y_opt = df.groupby("id")["sales"].shift(-h_opt)
valid_h = y_opt.notna()

# Optuna uses a smaller time window for speed (d_1600..d_1885-h, val d_1886-h..d_1913-h)
opt_tr_mask = (
    (df["d_num"] >= OPTUNA_START) &
    (df["d_num"] <= VAL_START - h_opt - 1) & valid_h
)
opt_vl_mask = (
    (df["d_num"] >= VAL_START - h_opt) &
    (df["d_num"] <= LAST_TRAIN - h_opt) & valid_h
)

X_opt_tr = df.loc[opt_tr_mask, ALL_FEATURES]
y_opt_tr  = y_opt[opt_tr_mask].astype(np.float32)
X_opt_vl  = df.loc[opt_vl_mask, ALL_FEATURES]
y_opt_vl  = y_opt[opt_vl_mask].astype(np.float32)
print(f"    Optuna train: {len(X_opt_tr):,}  val: {len(X_opt_vl):,}")

ds_opt_tr = lgb.Dataset(X_opt_tr, label=y_opt_tr, categorical_feature=CAT_FEATURES, free_raw_data=False)
ds_opt_vl = lgb.Dataset(X_opt_vl, label=y_opt_vl, categorical_feature=CAT_FEATURES, reference=ds_opt_tr, free_raw_data=False)

def optuna_objective(trial):
    p = {
        "objective":              "tweedie",
        "metric":                 "tweedie",
        "verbose":               -1,
        "seed":                   42,
        "num_threads":            0,
        "bagging_freq":           1,
        "learning_rate":          trial.suggest_categorical("lr",        [0.025, 0.05, 0.075, 0.1]),
        "num_leaves":             trial.suggest_categorical("num_leaves", [32, 64, 128, 256]),
        "min_data_in_leaf":       trial.suggest_categorical("min_data",   [20, 50, 100, 200]),
        "feature_fraction":       trial.suggest_categorical("feat_frac",  [0.5, 0.7, 0.8, 0.9]),
        "bagging_fraction":       trial.suggest_categorical("bag_frac",   [0.5, 0.7, 0.8, 0.9]),
        "lambda_l2":              trial.suggest_categorical("l2",         [0.0, 0.1, 0.5, 1.0]),
        "tweedie_variance_power": trial.suggest_float("tvp", 1.0, 1.9),
    }
    m = lgb.train(
        p, ds_opt_tr, num_boost_round=500,
        valid_sets=[ds_opt_vl],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
    )
    return m.best_score["valid_0"]["tweedie"]

study = optuna.create_study(direction="minimize")
t0 = time.time()
study.optimize(optuna_objective, n_trials=N_TRIALS, show_progress_bar=False)
optuna_time = time.time() - t0
bp = study.best_params
print(f"    Done in {optuna_time:.0f}s  Best: {study.best_value:.4f}")
print(f"    Best params: {bp}")

del ds_opt_tr, ds_opt_vl, X_opt_tr, y_opt_tr, X_opt_vl, y_opt_vl, y_opt
gc.collect()

BEST_PARAMS = {
    "objective":              "tweedie",
    "metric":                 "tweedie",
    "verbose":               -1,
    "num_threads":            0,
    "seed":                   42,
    "bagging_freq":           1,
    "learning_rate":          bp.get("lr",        0.05),
    "num_leaves":             bp.get("num_leaves", 64),
    "min_data_in_leaf":       bp.get("min_data",   50),
    "feature_fraction":       bp.get("feat_frac",  0.9),
    "bagging_fraction":       bp.get("bag_frac",   0.9),
    "lambda_l2":              bp.get("l2",         1.0),
    "tweedie_variance_power": bp.get("tvp",        1.5),
}

# ── 4. Train all 28 models ────────────────────────────────────────────────────
print(f"\n[4] Training {HORIZON} models (h=1..{HORIZON})...")
models = {}
h_scores = {}  # {h: {"best_iter": int, "val_tweedie": float, "train_s": float}}
t_total = time.time()

for h in range(1, HORIZON + 1):
    path = os.path.join(MH_DIR, f"h_{h:02d}.pkl")
    if os.path.exists(path):
        # Resume: load already-trained model
        with open(path, "rb") as f:
            models[h] = pickle.load(f)
        print(f"  h={h:2d}: loaded from checkpoint  {path}")
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

    val_metric = model.best_score["valid_0"]["tweedie"]
    h_scores[h] = {
        "best_iter":   model.best_iteration,
        "val_tweedie": float(val_metric),
        "train_s":     float(elapsed),
    }
    print(f"  h={h:2d}: iter={model.best_iteration:4d}  val={val_metric:.4f}  {elapsed:.0f}s  saved")
    models[h] = model

    del ds_tr, ds_vl, X_tr, y_tr, X_vl, y_vl, y_h
    gc.collect()

print(f"\n    All {HORIZON} models trained in {(time.time()-t_total)/60:.1f} min")

# ── 5. Validation WRMSSE from origin d_1913 ──────────────────────────────────
print(f"\n[5] Val WRMSSE (multi-horizon from origin d_{LAST_TRAIN})...")
val_preds_mh = np.zeros((n_series, HORIZON), dtype=np.float32)
t0 = time.time()
for h in range(1, HORIZON + 1):
    p = np.clip(models[h].predict(df_val_origin[ALL_FEATURES]), 0.0, None).astype(np.float32)
    val_preds_mh[:, h-1] = p
print(f"    Inference done in {time.time()-t0:.1f}s")

wrmsse_mh_val = score_preds(val_preds_mh, series_ids)
print(f"\n    Multi-horizon WRMSSE (origin d_{LAST_TRAIN}): {wrmsse_mh_val:.4f}")
print(f"    Single-step WRMSSE (oracle, Day 6/7):        0.5422  [reference]")
print(f"    Recursive WRMSSE (Day 8 v2):                 0.6019  [reference]")
print(f"    Gap vs oracle:      {wrmsse_mh_val - 0.5422:+.4f}")
print(f"    Gap vs recursive:   {wrmsse_mh_val - 0.6019:+.4f}")

# Per-horizon absolute error table
actual_mat = (
    sales_eval[sales_eval["id"].isin(series_ids)]
    .set_index("id").reindex(series_ids)[actual_val_cols]
    .values.astype(np.float32)
)
per_h_mae = [float(np.abs(val_preds_mh[:, h-1] - actual_mat[:, h-1]).mean()) for h in range(1, HORIZON+1)]
print(f"\n    Per-horizon MAE (origin d_{LAST_TRAIN}):")
print(f"    {'h':>3}  {'MAE':>8}  |  {'h':>3}  {'MAE':>8}  |  {'h':>3}  {'MAE':>8}  |  {'h':>3}  {'MAE':>8}")
for i in range(7):
    cols = []
    for j in range(4):
        idx = i + j * 7
        if idx < HORIZON:
            cols.append(f"  {idx+1:3d}  {per_h_mae[idx]:8.4f}")
        else:
            cols.append("              ")
    print("   " + " |".join(cols))

# ── 6. Eval predictions from origin d_1941 ────────────────────────────────────
print(f"\n[6] Eval predictions (multi-horizon from origin d_{LAST_TRAIN + HORIZON})...")
eval_preds_mh = np.zeros((n_series, HORIZON), dtype=np.float32)
t0 = time.time()
for h in range(1, HORIZON + 1):
    p = np.clip(models[h].predict(df_eval_origin[ALL_FEATURES]), 0.0, None).astype(np.float32)
    eval_preds_mh[:, h-1] = p
print(f"    Eval inference done in {time.time()-t0:.1f}s")

# ── 7. Load single-step val predictions (for submission val rows) ─────────────
print(f"\n[7] Loading single-step val predictions (d_{LAST_TRAIN+1}-{LAST_TRAIN+HORIZON})...")
df_ss = pd.read_parquet(
    FEAT_DIR,
    filters=[("d_num", ">=", LAST_TRAIN + 1), ("d_num", "<=", LAST_TRAIN + HORIZON)],
    columns=["id", "d_num"] + ALL_FEATURES,
)
for col, dtype in CAT_DTYPES.items():
    if col in df_ss.columns:
        df_ss[col] = df_ss[col].astype(dtype)
df_ss = df_ss.sort_values(["id", "d_num"]).reset_index(drop=True)

with open(os.path.join(MODELS, "lgbm_best.pkl"), "rb") as _f:
    model_global = pickle.load(_f)

preds_ss_flat = np.clip(model_global.predict(df_ss[ALL_FEATURES]), 0, None).astype(np.float32)
val_preds_ss  = preds_ss_flat.reshape(n_series, HORIZON)
wrmsse_ss = score_preds(val_preds_ss, series_ids)
print(f"    Single-step WRMSSE (check): {wrmsse_ss:.4f}")

# ── 8. Build Kaggle submission ─────────────────────────────────────────────────
print(f"\n[8] Building submission...")
sn28 = pd.read_csv(os.path.join(SUBS, "seasonal_naive_28_submission.csv"))
f_cols = [f"F{i}" for i in range(1, HORIZON + 1)]

def build_submission(val_mat, eval_mat, sids, fname):
    base = sn28.copy().set_index("id")
    val_ids  = [s.replace("_evaluation", "_validation") for s in sids]
    val_df   = pd.DataFrame(val_mat, columns=f_cols)
    val_df.insert(0, "id", val_ids)
    val_idx  = pd.Index(val_ids).intersection(base.index)
    base.loc[val_idx] = val_df.set_index("id").loc[val_idx]
    eval_df  = pd.DataFrame(eval_mat, columns=f_cols)
    eval_df.insert(0, "id", list(sids))
    eval_idx = pd.Index(sids).intersection(base.index)
    base.loc[eval_idx] = eval_df.set_index("id").loc[eval_idx]
    path = os.path.join(SUBS, fname)
    base.reset_index().to_csv(path, index=False)
    print(f"    Saved: {fname}  ({len(val_idx)} val + {len(eval_idx)} eval rows)")
    return path

# Main submission: single-step val + multi-horizon eval
build_submission(val_preds_ss, eval_preds_mh, series_ids, "mh_global.csv")

# Blend: single-step val, blend eval (multi-horizon + Day 7 recursive)
# Blend: MH eval + Day 8 blend eval for ensemble diversity
try:
    day8_blend = pd.read_csv(os.path.join(SUBS, "lgbm_v2_blend.csv")).set_index("id")
    eval_ids_list = list(series_ids)   # _evaluation suffix IDs
    day8_eval_mat = day8_blend.loc[eval_ids_list, f_cols].values.astype(np.float32)
    eval_blend_mat = 0.5 * eval_preds_mh + 0.5 * day8_eval_mat
    build_submission(val_preds_ss, eval_blend_mat, series_ids, "mh_blend.csv")
    print(f"    mh_blend.csv = 0.5 x multi-horizon + 0.5 x Day8-blend (eval rows only)")
except Exception as e:
    print(f"    Skipping blend submission: {e}")

# ── 9. Save results JSON ───────────────────────────────────────────────────────
def make_serial(obj):
    if isinstance(obj, (np.floating, float)): return float(obj)
    if isinstance(obj, (np.integer, int)):    return int(obj)
    if isinstance(obj, dict):  return {k: make_serial(v) for k, v in obj.items()}
    if isinstance(obj, list):  return [make_serial(x) for x in obj]
    return obj

results = {
    "optuna": {
        "horizon_tuned": OPTUNA_H,
        "n_trials": N_TRIALS,
        "best_value": float(study.best_value),
        "best_params": bp,
    },
    "final_params": {k: float(v) if isinstance(v, float) else v for k, v in BEST_PARAMS.items()},
    "val_wrmsse": {
        "multi_horizon_from_d1913": wrmsse_mh_val,
        "single_step_oracle":       wrmsse_ss,
        "recursive_v2_reference":   0.6019,
    },
    "per_horizon_mae": {str(h): per_h_mae[h-1] for h in range(1, HORIZON+1)},
    "per_horizon_training": {str(h): h_scores.get(h, {}) for h in range(1, HORIZON+1)},
}

scores_path = os.path.join(REPORTS, "day9_scores.json")
with open(scores_path, "w") as f:
    json.dump(make_serial(results), f, indent=2)
print(f"\n    Saved: {scores_path}")

# ── 10. Write report ───────────────────────────────────────────────────────────
print("\n[10] Writing reports/09_multi_horizon.md...")
report = f"""# Day 9 — Multi-Horizon LightGBM (28 Direct-Horizon Models)

## Overview

Train one LightGBM model per forecast horizon (h=1..28). Each model_h predicts
`sales[d+h]` given features available at time d. At inference, all 28 models
use a single origin point (d_{LAST_TRAIN + HORIZON} for eval), so no recursive
error compounding is possible.

---

## Training Design

| Component | Configuration |
|-----------|---------------|
| Models | 28 (one per horizon h=1..28) |
| Features | Same 38 features as Day 6 global (all available at time d) |
| Target | `df.groupby("id")["sales"].shift(-h)` |
| Train rows | `d_num in [FEAT_START, VAL_START - h - 1]` |
| Val rows | `d_num in [VAL_START - h, LAST_TRAIN - h]` (targets in d_1886..d_1913) |
| Hyperparameters | Optuna {N_TRIALS} trials on h={OPTUNA_H}, shared across all h |
| Optuna best params | lr={bp.get('lr')} leaves={bp.get('num_leaves')} tvp={bp.get('tvp', 0):.3f} |
| Inference origin | d_{LAST_TRAIN + HORIZON} (all 28 models, actual sales features, no recursion) |

---

## Optuna Results (h={OPTUNA_H})

- Best Tweedie val loss: {study.best_value:.4f}
- tvp: {bp.get('tvp', 0):.3f}
- lr: {bp.get('lr')}  num_leaves: {bp.get('num_leaves')}

---

## Validation WRMSSE (origin d_{LAST_TRAIN})

| Method | WRMSSE | Gap vs oracle |
|--------|--------|---------------|
| Single-step oracle (Day 6/7, actual features each day) | 0.5422 | 0 (reference) |
| **Multi-horizon from origin d_{LAST_TRAIN}** | **{wrmsse_mh_val:.4f}** | **{wrmsse_mh_val - 0.5422:+.4f}** |
| Recursive v2 (Day 8) | 0.6019 | +0.0597 |

Multi-horizon val WRMSSE lies between oracle (uses actual features per day) and
recursive (compounds prediction error over 28 steps). The gap vs oracle remains
because features at d_{LAST_TRAIN} are 1-28 days stale for predicting d_{LAST_TRAIN+1}..d_{LAST_TRAIN+28}.

---

## Per-Horizon MAE

| h | MAE | h | MAE | h | MAE | h | MAE |
|---|-----|---|-----|---|-----|---|-----|
"""

for i in range(7):
    row = "|"
    for j in range(4):
        idx = i + j * 7
        if idx < HORIZON:
            row += f" {idx+1} | {per_h_mae[idx]:.4f} |"
        else:
            row += " — | — |"
    report += row + "\n"

report += f"""
---

## Key Insight: Why Multi-Horizon Should Win on Private LB

The private LB period (d_{EVAL_START}–d_{EVAL_END}) requires forecasting 28 days from d_{LAST_TRAIN + HORIZON}.

- **Recursive**: d_{EVAL_END} prediction uses 27 compounded predicted lag features (noisy)
- **Multi-horizon**: d_{EVAL_END} prediction uses model_28 on d_{LAST_TRAIN + HORIZON}'s actual features (clean)

For HOBBIES (77% zeros), recursive compounding causes near-zero predictions to dominate lag
features, leading to persistent under-prediction. Multi-horizon model_28 was trained
specifically on 28-step-ahead patterns and has seen many (d, d+28) training pairs,
learning the typical 28-day-ahead demand distribution directly.

---

## Files

| File | Description |
|------|-------------|
| `scripts/09_train_multi_horizon.py` | Full training + inference pipeline |
| `data/models/multi_horizon/h_{{01..28}}.pkl` | 28 trained models (gitignored) |
| `reports/day9_scores.json` | Full results + per-horizon stats |
| `submissions/mh_global.csv` | Val: single-step. Eval: multi-horizon global. Submit this. |
| `submissions/mh_blend.csv` | Val: single-step. Eval: 0.5×MH + 0.5×Day8 blend |
"""

with open(os.path.join(REPORTS, "09_multi_horizon.md"), "w", encoding="utf-8") as f:
    f.write(report)
print("    Saved: reports/09_multi_horizon.md")

# ── summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("DAY 9 SUMMARY")
print("=" * 60)
print(f"\nOptuna (h={OPTUNA_H}, {N_TRIALS} trials):  best={study.best_value:.4f}")
print(f"Final params:  lr={BEST_PARAMS['learning_rate']}  leaves={BEST_PARAMS['num_leaves']}  tvp={BEST_PARAMS['tweedie_variance_power']:.3f}")
print(f"\nVal WRMSSE (multi-horizon from origin d_{LAST_TRAIN}):")
print(f"  Multi-horizon: {wrmsse_mh_val:.4f}")
print(f"  Oracle (ref):  0.5422")
print(f"  Recursive ref: 0.6019")
print(f"  vs oracle: {wrmsse_mh_val-0.5422:+.4f}  vs recursive: {wrmsse_mh_val-0.6019:+.4f}")
print(f"\nSubmissions built:")
print(f"  submissions/mh_global.csv   — multi-horizon eval (RECOMMEND SUBMIT)")
print(f"  submissions/mh_blend.csv    — 0.5×MH + 0.5×Day8 blend (submit second)")
print(f"\nTo submit:")
print(f'  kaggle competitions submit -c m5-forecasting-accuracy -f submissions/mh_global.csv -m "Day 9: multi-horizon 28 direct models"')
print("\nDone!")
