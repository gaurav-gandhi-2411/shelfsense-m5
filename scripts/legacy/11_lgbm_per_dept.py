"""
WS2.5 Variant 1 — Per-Department Multi-Horizon LightGBM

Trains 7 dept-specific models (FOODS_1..3, HOUSEHOLD_1..2, HOBBIES_1..2).
Each dept model = 28 horizon-specific LightGBMs, same architecture as Day 9.

Rationale: M5 departments have structurally different demand distributions:
  FOODS (~50% WRMSSE weight, 62% zero rate)      — high volume, strong weekly cycle
  HOUSEHOLD (~30% weight, 72% zero rate)          — mid volume, promo-driven spikes
  HOBBIES (~20% weight, 77% zero rate)            — sparse intermittent demand
A global model must compromise; per-dept models optimise each distribution separately.
This mirrors the M5 1st-place winner's per-dept/per-store decomposition strategy.

Generates:
  data/models/per_dept/{dept_id}/h_{h:02d}.pkl   — 7 × 28 = 196 models
  data/predictions/lgbm_per_dept_eval.parquet     — eval preds (n_series × 28)
  data/predictions/lgbm_per_dept_val.parquet      — val preds  (n_series × 28)
  submissions/lgbm_per_dept.csv                   — Kaggle submission
  reports/day11_per_dept_scores.json              — per-dept WRMSSE breakdown
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
print("WS2.5 Variant 1 — Per-Dept Multi-Horizon LightGBM (7 × 28 models)")
print("=" * 65)

# ── paths ─────────────────────────────────────────────────────────────────────
DATA_RAW  = os.path.join(PROJ_ROOT, "data", "raw", "m5-forecasting-accuracy")
FEAT_DIR  = os.path.join(PROJ_ROOT, "data", "processed", "features")
MODELS    = os.path.join(PROJ_ROOT, "data", "models", "per_dept")
PREDS_DIR = os.path.join(PROJ_ROOT, "data", "predictions")
REPORTS   = os.path.join(PROJ_ROOT, "reports")
SUBS      = os.path.join(PROJ_ROOT, "submissions")
os.makedirs(PREDS_DIR, exist_ok=True)

# ── constants (match Day 9 exactly) ───────────────────────────────────────────
LAST_TRAIN   = 1913
VAL_START    = 1886
FEAT_START   = 1000
OPTUNA_START = 1600
HORIZON      = 28
N_TRIALS     = 10          # per dept (faster than Day 9's 15-trial global search)
OPTUNA_H     = 14

DEPTS = ["FOODS_1", "FOODS_2", "FOODS_3", "HOUSEHOLD_1", "HOUSEHOLD_2", "HOBBIES_1", "HOBBIES_2"]

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

# Day 9 best params as warm-start defaults (avoids full cold-start Optuna)
DAY9_DEFAULTS = {
    "learning_rate":          0.025,
    "num_leaves":             64,
    "min_data_in_leaf":       100,
    "feature_fraction":       0.7,
    "bagging_fraction":       0.9,
    "lambda_l2":              0.1,
    "tweedie_variance_power": 1.5316,
}

# ── 1. Load full training data once ───────────────────────────────────────────
print(f"\n[1] Loading training data (d_num {FEAT_START}–{LAST_TRAIN})...")
t0 = time.time()
df_all = pd.read_parquet(
    FEAT_DIR,
    filters=[("d_num", ">=", FEAT_START), ("d_num", "<=", LAST_TRAIN)],
    columns=["id", "item_id", "cat_id", "dept_id", "store_id", "state_id",
             "d_num", "sales"] + NUM_FEATURES,
)
for col, dtype in CAT_DTYPES.items():
    if col in df_all.columns:
        df_all[col] = df_all[col].astype(dtype)
df_all = df_all.dropna(subset=["lag_7", "lag_14", "lag_28", "lag_56"]).reset_index(drop=True)
print(f"    Rows: {len(df_all):,}  ({time.time()-t0:.1f}s)")

print("\n[1b] Loading raw CSVs...")
t0 = time.time()
sales_eval  = pd.read_csv(os.path.join(DATA_RAW, "sales_train_evaluation.csv"))
prices_df   = pd.read_csv(os.path.join(DATA_RAW, "sell_prices.csv"))
calendar_df = pd.read_csv(os.path.join(DATA_RAW, "calendar.csv"))
print(f"     Loaded in {time.time()-t0:.1f}s")

# ── 2. Load inference origins (all series, both origins) ──────────────────────
print(f"\n[2] Loading inference origins (d_{LAST_TRAIN} and d_{LAST_TRAIN + HORIZON})...")
df_origins_all = pd.read_parquet(
    FEAT_DIR,
    filters=[("d_num", ">=", LAST_TRAIN), ("d_num", "<=", LAST_TRAIN + HORIZON)],
    columns=["id", "item_id", "cat_id", "dept_id", "store_id", "state_id",
             "d_num"] + NUM_FEATURES,
)
for col, dtype in CAT_DTYPES.items():
    if col in df_origins_all.columns:
        df_origins_all[col] = df_origins_all[col].astype(dtype)

df_val_origin_all  = df_origins_all[df_origins_all["d_num"] == LAST_TRAIN].sort_values("id").reset_index(drop=True)
df_eval_origin_all = df_origins_all[df_origins_all["d_num"] == LAST_TRAIN + HORIZON].sort_values("id").reset_index(drop=True)
all_series_ids = df_val_origin_all["id"].values
print(f"    Total series: {len(all_series_ids):,}")

f_cols = [f"F{i}" for i in range(1, HORIZON + 1)]
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

# ── Accumulators for full-submission assembly ──────────────────────────────────
all_val_preds  = {}   # {id: [F1..F28]}
all_eval_preds = {}

dept_results = {}

# ── 3. Per-dept training loop ──────────────────────────────────────────────────
t_ws25_start = time.time()

for dept_id in DEPTS:
    dept_dir = os.path.join(MODELS, dept_id)
    os.makedirs(dept_dir, exist_ok=True)

    print(f"\n{'-'*65}")
    print(f"DEPT: {dept_id}")
    print(f"{'-'*65}")

    # Filter training data to this dept
    df = df_all[df_all["dept_id"] == dept_id].copy()
    n_dept_series = df["id"].nunique()
    print(f"  Series: {n_dept_series}  |  Rows: {len(df):,}")

    # Filter inference origins
    dept_val_origin  = df_val_origin_all[df_val_origin_all["dept_id"] == dept_id].sort_values("id").reset_index(drop=True)
    dept_eval_origin = df_eval_origin_all[df_eval_origin_all["dept_id"] == dept_id].sort_values("id").reset_index(drop=True)
    dept_ids = dept_val_origin["id"].values

    # ── Optuna: per-dept search (10 trials on h=14) ──────────────────────────
    print(f"\n  [Optuna] {N_TRIALS} trials on h={OPTUNA_H} for {dept_id}...")
    h_opt    = OPTUNA_H
    y_opt    = df.groupby("id")["sales"].shift(-h_opt)
    valid_h  = y_opt.notna()

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
    print(f"  Optuna train: {len(X_opt_tr):,}  val: {len(X_opt_vl):,}")

    ds_opt_tr = lgb.Dataset(X_opt_tr, label=y_opt_tr, categorical_feature=CAT_FEATURES, free_raw_data=False)
    ds_opt_vl = lgb.Dataset(X_opt_vl, label=y_opt_vl, categorical_feature=CAT_FEATURES, reference=ds_opt_tr, free_raw_data=False)

    def optuna_objective(trial):
        p = {
            "objective": "tweedie", "metric": "tweedie",
            "verbose": -1, "seed": 42, "num_threads": 0, "bagging_freq": 1,
            "learning_rate":          trial.suggest_categorical("lr",        [0.025, 0.05, 0.1]),
            "num_leaves":             trial.suggest_categorical("num_leaves", [32, 64, 128]),
            "min_data_in_leaf":       trial.suggest_categorical("min_data",   [20, 50, 100]),
            "feature_fraction":       trial.suggest_categorical("feat_frac",  [0.7, 0.8, 0.9]),
            "bagging_fraction":       trial.suggest_categorical("bag_frac",   [0.7, 0.8, 0.9]),
            "lambda_l2":              trial.suggest_categorical("l2",         [0.0, 0.1, 0.5]),
            "tweedie_variance_power": trial.suggest_float("tvp", 1.0, 1.9),
        }
        m = lgb.train(
            p, ds_opt_tr, num_boost_round=500,
            valid_sets=[ds_opt_vl],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
        )
        return m.best_score["valid_0"]["tweedie"]

    study = optuna.create_study(direction="minimize")
    t_opt = time.time()
    study.optimize(optuna_objective, n_trials=N_TRIALS, show_progress_bar=False)
    bp = study.best_params
    print(f"  Optuna done in {time.time()-t_opt:.0f}s  best={study.best_value:.4f}")
    print(f"  Best: lr={bp.get('lr')} leaves={bp.get('num_leaves')} tvp={bp.get('tvp',0):.3f}")

    del ds_opt_tr, ds_opt_vl, X_opt_tr, y_opt_tr, X_opt_vl, y_opt_vl, y_opt
    gc.collect()

    BEST_PARAMS = {
        "objective": "tweedie", "metric": "tweedie",
        "verbose": -1, "num_threads": 0, "seed": 42, "bagging_freq": 1,
        "learning_rate":          bp.get("lr",        DAY9_DEFAULTS["learning_rate"]),
        "num_leaves":             bp.get("num_leaves", DAY9_DEFAULTS["num_leaves"]),
        "min_data_in_leaf":       bp.get("min_data",   DAY9_DEFAULTS["min_data_in_leaf"]),
        "feature_fraction":       bp.get("feat_frac",  DAY9_DEFAULTS["feature_fraction"]),
        "bagging_fraction":       bp.get("bag_frac",   DAY9_DEFAULTS["bagging_fraction"]),
        "lambda_l2":              bp.get("l2",         DAY9_DEFAULTS["lambda_l2"]),
        "tweedie_variance_power": bp.get("tvp",        DAY9_DEFAULTS["tweedie_variance_power"]),
    }

    # ── Train 28 horizon models ───────────────────────────────────────────────
    print(f"\n  Training 28 models for {dept_id}...")
    models = {}
    h_scores = {}
    t_dept = time.time()

    for h in range(1, HORIZON + 1):
        path = os.path.join(dept_dir, f"h_{h:02d}.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                models[h] = pickle.load(f)
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
        X_vl = df.loc[val_mask,   ALL_FEATURES]
        y_vl = y_h[val_mask].astype(np.float32)

        ds_tr = lgb.Dataset(X_tr, label=y_tr.values, categorical_feature=CAT_FEATURES, free_raw_data=False)
        ds_vl = lgb.Dataset(X_vl, label=y_vl.values, categorical_feature=CAT_FEATURES, reference=ds_tr, free_raw_data=False)

        t0 = time.time()
        model = lgb.train(
            BEST_PARAMS, ds_tr, num_boost_round=3000,
            valid_sets=[ds_vl],
            callbacks=[lgb.early_stopping(75, verbose=False), lgb.log_evaluation(-1)],
        )
        elapsed = time.time() - t0

        with open(path, "wb") as f:
            pickle.dump(model, f)

        h_scores[h] = {
            "best_iter":   model.best_iteration,
            "val_tweedie": float(model.best_score["valid_0"]["tweedie"]),
            "train_s":     float(elapsed),
        }
        models[h] = model

        del ds_tr, ds_vl, X_tr, y_tr, X_vl, y_vl, y_h
        gc.collect()

    dept_train_time = time.time() - t_dept
    print(f"  {dept_id}: 28 models in {dept_train_time/60:.1f} min")

    # ── Inference: val + eval ─────────────────────────────────────────────────
    val_preds_dept  = np.zeros((len(dept_ids), HORIZON), dtype=np.float32)
    eval_preds_dept = np.zeros((len(dept_ids), HORIZON), dtype=np.float32)

    for h in range(1, HORIZON + 1):
        val_preds_dept[:, h-1]  = np.clip(models[h].predict(dept_val_origin[ALL_FEATURES]),  0.0, None)
        eval_preds_dept[:, h-1] = np.clip(models[h].predict(dept_eval_origin[ALL_FEATURES]), 0.0, None)

    # ── Val WRMSSE for this dept ──────────────────────────────────────────────
    dept_wrmsse = score_preds(val_preds_dept, dept_ids)
    print(f"  Val WRMSSE ({dept_id}): {dept_wrmsse:.4f}")

    dept_results[dept_id] = {
        "n_series":       int(n_dept_series),
        "val_wrmsse":     dept_wrmsse,
        "optuna_best":    float(study.best_value),
        "best_params":    {k: float(v) if isinstance(v, float) else v for k, v in BEST_PARAMS.items()},
        "train_time_min": dept_train_time / 60,
        "h_scores":       h_scores,
    }

    # Accumulate for full-submission assembly
    for i, sid in enumerate(dept_ids):
        all_val_preds[sid]  = val_preds_dept[i]
        all_eval_preds[sid] = eval_preds_dept[i]

    del models, val_preds_dept, eval_preds_dept, df
    gc.collect()

total_time = time.time() - t_ws25_start
print(f"\n{'='*65}")
print(f"All 7 depts trained in {total_time/60:.1f} min")

# ── 4. Assemble full submission arrays ────────────────────────────────────────
print("\n[4] Assembling full-coverage predictions...")
# Preserve Day 9 ordering (all_series_ids from val_origin)
n_total = len(all_series_ids)
full_val_mat  = np.zeros((n_total, HORIZON), dtype=np.float32)
full_eval_mat = np.zeros((n_total, HORIZON), dtype=np.float32)
missing = []
for i, sid in enumerate(all_series_ids):
    if sid in all_val_preds:
        full_val_mat[i]  = all_val_preds[sid]
        full_eval_mat[i] = all_eval_preds[sid]
    else:
        missing.append(sid)

if missing:
    print(f"  WARNING: {len(missing)} series not covered by any dept model.")
    print(f"  Sample missing: {missing[:5]}")
else:
    print(f"  All {n_total:,} series covered (0 missing)")

# ── 5. Overall val WRMSSE ─────────────────────────────────────────────────────
print("\n[5] Overall val WRMSSE (per-dept ensemble)...")
overall_wrmsse = score_preds(full_val_mat, all_series_ids)
print(f"    Per-dept overall:      {overall_wrmsse:.4f}")
print(f"    Day 9 mh_blend (ref):  0.5854  (private LB, not val)")
print(f"    Single-step oracle:    0.5422  (reference)")

# Per-dept table
print("\n    Per-dept breakdown:")
print(f"    {'Dept':<15}  {'Series':>7}  {'Val WRMSSE':>11}")
for dept_id in DEPTS:
    r = dept_results[dept_id]
    print(f"    {dept_id:<15}  {r['n_series']:>7,}  {r['val_wrmsse']:>11.4f}")

# ── 6. Save predictions to parquet ───────────────────────────────────────────
print("\n[6] Saving prediction parquets...")
val_df  = pd.DataFrame(full_val_mat,  columns=f_cols)
val_df.insert(0, "id", all_series_ids)
val_df.to_parquet(os.path.join(PREDS_DIR, "lgbm_per_dept_val.parquet"), index=False)

eval_df = pd.DataFrame(full_eval_mat, columns=f_cols)
eval_df.insert(0, "id", all_series_ids)
eval_df.to_parquet(os.path.join(PREDS_DIR, "lgbm_per_dept_eval.parquet"), index=False)
print("    Saved lgbm_per_dept_val.parquet and lgbm_per_dept_eval.parquet")

# ── 7. Build Kaggle submission ────────────────────────────────────────────────
print("\n[7] Building submission...")
sn28     = pd.read_csv(os.path.join(SUBS, "seasonal_naive_28_submission.csv"))

def build_submission(val_mat, eval_mat, sids, fname):
    base     = sn28.copy().set_index("id")
    val_ids  = [s.replace("_evaluation", "_validation") for s in sids]
    val_df2  = pd.DataFrame(val_mat, columns=f_cols)
    val_df2.insert(0, "id", val_ids)
    val_idx  = pd.Index(val_ids).intersection(base.index)
    base.loc[val_idx] = val_df2.set_index("id").loc[val_idx]
    eval_df2 = pd.DataFrame(eval_mat, columns=f_cols)
    eval_df2.insert(0, "id", list(sids))
    eval_idx = pd.Index(sids).intersection(base.index)
    base.loc[eval_idx] = eval_df2.set_index("id").loc[eval_idx]
    path = os.path.join(SUBS, fname)
    base.reset_index().to_csv(path, index=False)
    print(f"    Saved {fname}  ({len(val_idx)} val + {len(eval_idx)} eval rows)")

build_submission(full_val_mat, full_eval_mat, all_series_ids, "lgbm_per_dept.csv")

# ── 8. Save results JSON ──────────────────────────────────────────────────────
def make_serial(obj):
    if isinstance(obj, (np.floating, float)): return float(obj)
    if isinstance(obj, (np.integer, int)):    return int(obj)
    if isinstance(obj, dict):  return {k: make_serial(v) for k, v in obj.items()}
    if isinstance(obj, list):  return [make_serial(x) for x in obj]
    return obj

results = {
    "variant": "lgbm_per_dept",
    "n_depts": len(DEPTS),
    "total_models": len(DEPTS) * HORIZON,
    "total_train_min": total_time / 60,
    "overall_val_wrmsse": overall_wrmsse,
    "reference_private_lb": 0.5854,
    "per_dept": make_serial(dept_results),
}

scores_path = os.path.join(REPORTS, "day11_per_dept_scores.json")
with open(scores_path, "w") as f:
    json.dump(make_serial(results), f, indent=2)
print(f"\n    Saved {scores_path}")

# ── summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("WS2.5 VARIANT 1 — SUMMARY")
print("=" * 65)
print(f"\nTotal training time: {total_time/60:.1f} min  ({len(DEPTS)*HORIZON} models)")
print(f"\nVal WRMSSE comparison:")
print(f"  Per-dept ensemble:  {overall_wrmsse:.4f}")
print(f"  Single-step oracle: 0.5422  (reference)")
print(f"\nPer-dept breakdown:")
for dept_id in DEPTS:
    r = dept_results[dept_id]
    print(f"  {dept_id:<15}  {r['val_wrmsse']:.4f}  ({r['n_series']:,} series, {r['train_time_min']:.1f} min)")
print(f"\nSubmission: submissions/lgbm_per_dept.csv")
print(f"\nTo submit:")
print(f'  kaggle competitions submit -c m5-forecasting-accuracy \\')
print(f'    -f submissions/lgbm_per_dept.csv \\')
print(f'    -m "WS2.5 Variant 1: per-dept multi-horizon LightGBM (7 depts x 28 horizons)"')
print("\nDone!")
