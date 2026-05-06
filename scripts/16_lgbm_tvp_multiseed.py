"""
WS2.5 — tvp=1.3 Multi-Seed Averaging (5 seeds)

Same architecture as scripts/12 tvp=1.3 run. Only seed varies across
{42, 7, 123, 2024, 31337}. All other params identical (Day 9 Optuna best).

Seed=42 reuses existing lgbm_tvp_1p3_{val,eval}.parquet — deterministic,
retraining gives bit-identical result. Saves ~114 min.
Seeds 7/123/2024/31337 trained fresh (~7.6 hours sequential).

Cache-safe naming (F19.5):
  data/models/tvp_1p3_seed_{seed}/h_01..28.pkl  (distinct dir per seed)
  data/predictions/lgbm_tvp_1p3_s{seed}_val.parquet
  data/predictions/lgbm_tvp_1p3_s{seed}_eval.parquet

After all seeds:
  a. Per-seed same-origin val WRMSSE
  b. Average eval + val preds → score averaged val WRMSSE
  c. Seed-to-seed variance summary
  d. Optuna 50-trial ensemble {multiseed_avg, RMSE-MH}
  e. Submission CSV if avg val improvement > 0.003 vs single seed=42 (0.6860)
"""
from __future__ import annotations
import sys, os, time, json, pickle, gc, shutil, warnings
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
print("WS2.5 -- tvp=1.3 Multi-Seed Averaging (5 seeds)")
print("=" * 65)

# ── paths ─────────────────────────────────────────────────────────────────────
DATA_RAW  = os.path.join(PROJ_ROOT, "data", "raw", "m5-forecasting-accuracy")
FEAT_DIR  = os.path.join(PROJ_ROOT, "data", "processed", "features")
PREDS_DIR = os.path.join(PROJ_ROOT, "data", "predictions")
REPORTS   = os.path.join(PROJ_ROOT, "reports")
SUBS      = os.path.join(PROJ_ROOT, "submissions")
MODELS    = os.path.join(PROJ_ROOT, "data", "models")

LAST_TRAIN  = 1913
VAL_START   = 1886
FEAT_START  = 1000
HORIZON     = 28
SINGLE_SEED_BASELINE = 0.6860   # seed=42 same-origin val WRMSSE
SUBMIT_THRESHOLD     = 0.003    # min improvement over baseline to submit

SEEDS = [42, 7, 123, 2024, 31337]

# Identical to scripts/12 — no yearly lags (ensures comparability with seed=42 run)
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

BASE_PARAMS = {
    "objective":              "tweedie",
    "metric":                 "tweedie",
    "verbose":               -1,
    "num_threads":            0,
    "bagging_freq":           1,
    "learning_rate":          0.025,
    "num_leaves":             64,
    "min_data_in_leaf":       100,
    "feature_fraction":       0.7,
    "bagging_fraction":       0.9,
    "lambda_l2":              0.1,
    "tweedie_variance_power": 1.3,
}

f_cols          = [f"F{i}" for i in range(1, HORIZON + 1)]
actual_val_cols = [f"d_{LAST_TRAIN + h}" for h in range(1, HORIZON + 1)]

print(f"\n  Seeds: {SEEDS}")
print(f"  Seed=42 will reuse existing parquets (deterministic)")
print(f"  Seeds {SEEDS[1:]} will train fresh (~{len(SEEDS)-1} x 114 min)")

# ── 1. Load training data (once, reused across all seeds) ─────────────────────
print(f"\n[1] Loading training data (d_num {FEAT_START}-{LAST_TRAIN})...")
t0 = time.time()
df = pd.read_parquet(
    FEAT_DIR,
    filters=[("d_num", ">=", FEAT_START), ("d_num", "<=", LAST_TRAIN)],
    columns=["id", "item_id", "cat_id", "dept_id", "store_id", "state_id",
             "d_num", "sales"] + NUM_FEATURES,
)
for col, dt in CAT_DTYPES.items():
    if col in df.columns:
        df[col] = df[col].astype(dt)
df = df.dropna(subset=["lag_7", "lag_14", "lag_28", "lag_56"]).reset_index(drop=True)
print(f"    Rows: {len(df):,}  ({time.time()-t0:.1f}s)")

# ── 1b. Raw CSVs for scoring ───────────────────────────────────────────────────
print("\n[1b] Loading raw CSVs...")
t0 = time.time()
sales_eval  = pd.read_csv(os.path.join(DATA_RAW, "sales_train_evaluation.csv"))
prices_df   = pd.read_csv(os.path.join(DATA_RAW, "sell_prices.csv"))
calendar_df = pd.read_csv(os.path.join(DATA_RAW, "calendar.csv"))
print(f"     Loaded in {time.time()-t0:.1f}s")

# ── 2. Load inference origins (once) ─────────────────────────────────────────
print(f"\n[2] Loading inference origins (d_{LAST_TRAIN} and d_{LAST_TRAIN+HORIZON})...")
df_origins = pd.read_parquet(
    FEAT_DIR,
    filters=[("d_num", ">=", LAST_TRAIN), ("d_num", "<=", LAST_TRAIN + HORIZON)],
    columns=["id", "cat_id", "dept_id", "store_id", "state_id", "d_num"] + NUM_FEATURES,
)
for col, dt in CAT_DTYPES.items():
    if col in df_origins.columns:
        df_origins[col] = df_origins[col].astype(dt)

df_val_origin  = df_origins[df_origins["d_num"] == LAST_TRAIN].sort_values("id").reset_index(drop=True)
df_eval_origin = df_origins[df_origins["d_num"] == LAST_TRAIN + HORIZON].sort_values("id").reset_index(drop=True)
series_ids = df_val_origin["id"].values
n_series   = len(series_ids)
val_ids    = [s.replace("_evaluation", "_validation") for s in series_ids]
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

# ── 3. Seed=42: copy existing parquets ────────────────────────────────────────
print("\n[3] Handling seed=42 (reuse existing parquets)...")
s42_val_path  = os.path.join(PREDS_DIR, "lgbm_tvp_1p3_s42_val.parquet")
s42_eval_path = os.path.join(PREDS_DIR, "lgbm_tvp_1p3_s42_eval.parquet")
src_val_path  = os.path.join(PREDS_DIR, "lgbm_tvp_1p3_val.parquet")
src_eval_path = os.path.join(PREDS_DIR, "lgbm_tvp_1p3_eval.parquet")

if not os.path.exists(s42_val_path):
    if os.path.exists(src_val_path):
        shutil.copy2(src_val_path,  s42_val_path)
        shutil.copy2(src_eval_path, s42_eval_path)
        print("    Copied lgbm_tvp_1p3_{val,eval}.parquet -> s42 variants")
    else:
        print("    WARNING: existing seed=42 parquets not found, will train fresh")
        SEEDS = SEEDS  # keep seed=42 in the training loop
else:
    print("    s42 parquets already exist, skipping")

# ── 4. Train seeds 7, 123, 2024, 31337 ────────────────────────────────────────
print("\n[4] Training fresh seeds...")
per_seed_wrmsse = {42: SINGLE_SEED_BASELINE}   # pre-populated from existing run
t_wall_start = time.time()

for seed in SEEDS:
    if seed == 42:
        continue   # handled above

    seed_dir = os.path.join(MODELS, f"tvp_1p3_seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)

    val_parquet  = os.path.join(PREDS_DIR, f"lgbm_tvp_1p3_s{seed}_val.parquet")
    eval_parquet = os.path.join(PREDS_DIR, f"lgbm_tvp_1p3_s{seed}_eval.parquet")

    # Check if this seed is fully cached (all 28 models + parquets exist)
    all_models_exist = all(
        os.path.exists(os.path.join(seed_dir, f"h_{h:02d}.pkl"))
        for h in range(1, HORIZON + 1)
    )
    if all_models_exist and os.path.exists(val_parquet) and os.path.exists(eval_parquet):
        print(f"\n  seed={seed}: fully cached, loading val WRMSSE from parquet...")
        vp = pd.read_parquet(val_parquet).set_index("id").loc[series_ids, f_cols].values.astype(np.float32)
        per_seed_wrmsse[seed] = score_preds(vp, series_ids)
        print(f"  seed={seed}: val WRMSSE = {per_seed_wrmsse[seed]:.4f}")
        continue

    print(f"\n  ---- seed={seed} ----")
    PARAMS = {**BASE_PARAMS, "seed": seed}

    models   = {}
    t_seed   = time.time()

    for h in range(1, HORIZON + 1):
        path = os.path.join(seed_dir, f"h_{h:02d}.pkl")
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
            PARAMS, ds_tr,
            num_boost_round=3000,
            valid_sets=[ds_vl],
            callbacks=[lgb.early_stopping(75, verbose=False), lgb.log_evaluation(500)],
        )
        elapsed = time.time() - t0

        with open(path, "wb") as f:
            pickle.dump(model, f)

        val_metric = float(model.best_score["valid_0"]["tweedie"])
        print(f"  s{seed} h={h:2d}: iter={model.best_iteration:4d}  val={val_metric:.4f}  {elapsed:.0f}s")
        models[h] = model

        del ds_tr, ds_vl, X_tr, y_tr, X_vl, y_vl, y_h
        gc.collect()

    seed_train_min = (time.time() - t_seed) / 60
    print(f"  seed={seed}: all 28 models in {seed_train_min:.1f} min")

    # Predict val and eval
    val_preds  = np.zeros((n_series, HORIZON), dtype=np.float32)
    eval_preds = np.zeros((n_series, HORIZON), dtype=np.float32)
    for h in range(1, HORIZON + 1):
        val_preds[:, h-1]  = np.clip(models[h].predict(df_val_origin[ALL_FEATURES]),  0.0, None)
        eval_preds[:, h-1] = np.clip(models[h].predict(df_eval_origin[ALL_FEATURES]), 0.0, None)

    # Score same-origin val WRMSSE
    wrmsse = score_preds(val_preds, series_ids)
    per_seed_wrmsse[seed] = wrmsse
    print(f"  seed={seed}: val WRMSSE = {wrmsse:.4f}  (baseline {SINGLE_SEED_BASELINE})")

    # Save parquets
    pd.DataFrame({"id": series_ids, **{f"F{h}": val_preds[:, h-1] for h in range(1, HORIZON+1)}}).to_parquet(val_parquet, index=False)
    pd.DataFrame({"id": series_ids, **{f"F{h}": eval_preds[:, h-1] for h in range(1, HORIZON+1)}}).to_parquet(eval_parquet, index=False)
    print(f"  seed={seed}: saved parquets")

    del models, val_preds, eval_preds
    gc.collect()

total_train_min = (time.time() - t_wall_start) / 60
print(f"\n  All fresh seeds done in {total_train_min:.1f} min")

# ── 5. Load all 5 seeds, average ──────────────────────────────────────────────
print("\n[5] Loading all 5 seed predictions and averaging...")

all_val_preds  = []
all_eval_preds = []

for seed in SEEDS:
    vp = pd.read_parquet(
        os.path.join(PREDS_DIR, f"lgbm_tvp_1p3_s{seed}_val.parquet")
    ).set_index("id").loc[series_ids, f_cols].values.astype(np.float32)
    ep = pd.read_parquet(
        os.path.join(PREDS_DIR, f"lgbm_tvp_1p3_s{seed}_eval.parquet")
    ).set_index("id").loc[series_ids, f_cols].values.astype(np.float32)
    all_val_preds.append(vp)
    all_eval_preds.append(ep)

val_avg  = np.mean(all_val_preds,  axis=0).astype(np.float32)
eval_avg = np.mean(all_eval_preds, axis=0).astype(np.float32)

# ── 6. Score: per-seed and averaged ──────────────────────────────────────────
print("\n[6] Scoring per-seed and averaged val WRMSSE...")
print(f"\n  Per-seed same-origin val WRMSSE:")
seed_scores = []
for seed in SEEDS:
    w = per_seed_wrmsse.get(seed)
    if w is None:
        vp = all_val_preds[SEEDS.index(seed)]
        w  = score_preds(vp, series_ids)
        per_seed_wrmsse[seed] = w
    seed_scores.append(w)
    print(f"    seed={seed:>5}: {w:.4f}")

wrmsse_avg = score_preds(val_avg, series_ids)
print(f"\n  Multi-seed average val WRMSSE: {wrmsse_avg:.4f}")
print(f"  Single-seed baseline (s42):    {SINGLE_SEED_BASELINE:.4f}")
improvement = SINGLE_SEED_BASELINE - wrmsse_avg
print(f"  Improvement:                   {improvement:+.4f}  (positive = better)")

seed_arr = np.array(seed_scores)
print(f"\n  Seed-to-seed variance summary:")
print(f"    min={seed_arr.min():.4f}  max={seed_arr.max():.4f}  "
      f"mean={seed_arr.mean():.4f}  std={seed_arr.std():.4f}  "
      f"range={seed_arr.max()-seed_arr.min():.4f}")

# Save avg parquets
avg_val_path  = os.path.join(PREDS_DIR, "lgbm_tvp_1p3_multiseed_avg_val.parquet")
avg_eval_path = os.path.join(PREDS_DIR, "lgbm_tvp_1p3_multiseed_avg_eval.parquet")
pd.DataFrame({"id": series_ids, **{f"F{h}": eval_avg[:, h-1] for h in range(1, HORIZON+1)}}).to_parquet(avg_eval_path, index=False)
pd.DataFrame({"id": series_ids, **{f"F{h}": val_avg[:, h-1]  for h in range(1, HORIZON+1)}}).to_parquet(avg_val_path, index=False)
print(f"\n  Saved multiseed avg parquets")

# ── 7. Optuna ensemble {multiseed_avg, RMSE-MH} ────────────────────────────────
print("\n[7] Optuna 50-trial ensemble {multiseed_avg, RMSE-MH}...")
rmse_val_path  = os.path.join(PREDS_DIR, "lgbm_rmse_mh_val.parquet")
rmse_eval_path = os.path.join(PREDS_DIR, "lgbm_rmse_mh_eval.parquet")

val_rmse  = pd.read_parquet(rmse_val_path).set_index("id").loc[series_ids, f_cols].values.astype(np.float32)
eval_rmse = pd.read_parquet(rmse_eval_path).set_index("id").loc[series_ids, f_cols].values.astype(np.float32)

wrmsse_rmse_standalone = score_preds(val_rmse, series_ids)
print(f"  RMSE-MH standalone val WRMSSE: {wrmsse_rmse_standalone:.4f}  (private LB was 0.6205)")

def optuna_objective(trial):
    w1 = trial.suggest_float("w_avg",  0.0, 1.0)
    w2 = trial.suggest_float("w_rmse", 0.0, 1.0)
    total = w1 + w2
    if total < 1e-6:
        return 9999.0
    blend = (w1 * val_avg + w2 * val_rmse) / total
    return score_preds(blend, series_ids)

study = optuna.create_study(direction="minimize")
t0 = time.time()
study.optimize(optuna_objective, n_trials=50, show_progress_bar=False)
opt_time = time.time() - t0

bp = study.best_params
total_w = bp["w_avg"] + bp["w_rmse"]
w_avg_n  = bp["w_avg"]  / total_w
w_rmse_n = bp["w_rmse"] / total_w

print(f"  Optuna done in {opt_time:.0f}s  best_val={study.best_value:.4f}")
print(f"  Optimal weights: multiseed_avg={w_avg_n:.3f}  RMSE-MH={w_rmse_n:.3f}")
print(f"  Improvement vs single seed=42: {SINGLE_SEED_BASELINE - study.best_value:+.4f}")
print(f"  Improvement vs multiseed_avg:  {wrmsse_avg - study.best_value:+.4f}")

# ── 8. Build submissions ───────────────────────────────────────────────────────
print("\n[8] Building submissions...")
sn28     = pd.read_csv(os.path.join(SUBS, "seasonal_naive_28_submission.csv"))
mh_sub   = pd.read_csv(os.path.join(SUBS, "mh_blend.csv")).set_index("id")
val_oracle_mat = mh_sub.loc[val_ids, f_cols].values.astype(np.float32)

def build_submission(eval_mat, fname):
    base = sn28.copy().set_index("id")
    vdf  = pd.DataFrame(val_oracle_mat, columns=f_cols)
    vdf.insert(0, "id", val_ids)
    val_idx  = pd.Index(val_ids).intersection(base.index)
    base.loc[val_idx] = vdf.set_index("id").loc[val_idx]
    edf  = pd.DataFrame(np.clip(eval_mat, 0.0, None).astype(np.float32), columns=f_cols)
    edf.insert(0, "id", list(series_ids))
    eval_idx = pd.Index(series_ids).intersection(base.index)
    base.loc[eval_idx] = edf.set_index("id").loc[eval_idx]
    path = os.path.join(SUBS, fname)
    base.reset_index().to_csv(path, index=False)
    print(f"  Saved {fname}")

build_submission(eval_avg, "lgbm_tvp_1p3_multiseed_avg.csv")

ensemble_built = False
ens_val_wrmsse = study.best_value
ens_improvement = SINGLE_SEED_BASELINE - study.best_value
if ens_improvement > SUBMIT_THRESHOLD:
    ens_eval = w_avg_n * eval_avg + w_rmse_n * eval_rmse
    build_submission(ens_eval, "ens_multiseed_rmse_optuna.csv")
    ensemble_built = True
    print(f"  Ensemble val improvement {ens_improvement:.4f} > {SUBMIT_THRESHOLD} -- CSV ready")
else:
    print(f"  Ensemble val improvement {ens_improvement:.4f} <= {SUBMIT_THRESHOLD} -- CSV not built")

# ── 9. Save results JSON ──────────────────────────────────────────────────────
def make_serial(obj):
    if isinstance(obj, (np.floating, float)): return float(obj)
    if isinstance(obj, (np.integer, int)):    return int(obj)
    if isinstance(obj, dict):  return {k: make_serial(v) for k, v in obj.items()}
    if isinstance(obj, list):  return [make_serial(x) for x in obj]
    return obj

results = {
    "seeds": SEEDS,
    "per_seed_val_wrmsse":   {str(s): per_seed_wrmsse[s] for s in SEEDS},
    "seed_stats": {
        "min": float(seed_arr.min()), "max": float(seed_arr.max()),
        "mean": float(seed_arr.mean()), "std": float(seed_arr.std()),
        "range": float(seed_arr.max() - seed_arr.min()),
    },
    "avg_val_wrmsse":       wrmsse_avg,
    "baseline_val_wrmsse":  SINGLE_SEED_BASELINE,
    "avg_improvement":      improvement,
    "optuna_ensemble": {
        "best_val": study.best_value,
        "weights":  {"w_avg": w_avg_n, "w_rmse": w_rmse_n},
        "improvement_vs_baseline": ens_improvement,
        "ensemble_csv_built": ensemble_built,
        "submit_threshold": SUBMIT_THRESHOLD,
    },
    "rmse_mh_standalone_val": wrmsse_rmse_standalone,
}
json_path = os.path.join(REPORTS, "day16_multiseed_scores.json")
with open(json_path, "w") as f:
    json.dump(make_serial(results), f, indent=2)
print(f"\n  Saved {json_path}")

# ── summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("MULTI-SEED SUMMARY")
print("=" * 65)
print(f"\n(a) Per-seed same-origin val WRMSSE:")
for seed in SEEDS:
    flag = "  (existing)" if seed == 42 else ""
    print(f"    seed={seed:>5}: {per_seed_wrmsse[seed]:.4f}{flag}")

print(f"\n(b) Averaged val WRMSSE: {wrmsse_avg:.4f}")
print(f"    Baseline (seed=42):  {SINGLE_SEED_BASELINE:.4f}")
print(f"    Improvement:         {improvement:+.4f}")

print(f"\n(c) Seed-to-seed variance:")
print(f"    min={seed_arr.min():.4f}  max={seed_arr.max():.4f}  "
      f"std={seed_arr.std():.4f}  range={seed_arr.max()-seed_arr.min():.4f}")

print(f"\n(d) Optuna ensemble {{multiseed_avg, RMSE-MH}}:")
print(f"    Best val WRMSSE:     {study.best_value:.4f}")
print(f"    Weights: avg={w_avg_n:.3f}  RMSE-MH={w_rmse_n:.3f}")
print(f"    Improvement vs s42:  {ens_improvement:+.4f}")

print(f"\n(e) Submissions:")
print(f"    lgbm_tvp_1p3_multiseed_avg.csv  (submit to get private LB for averaging)")
if ensemble_built:
    print(f"    ens_multiseed_rmse_optuna.csv   (ensemble -- improvement {ens_improvement:.4f} > {SUBMIT_THRESHOLD})")
    print(f"\n    Submit commands:")
    print(f"      kaggle competitions submit -c m5-forecasting-accuracy \\")
    print(f"        -f submissions/lgbm_tvp_1p3_multiseed_avg.csv \\")
    print(f"        -m \"WS2.5 tvp=1.3 5-seed average\"")
    print(f"      kaggle competitions submit -c m5-forecasting-accuracy \\")
    print(f"        -f submissions/ens_multiseed_rmse_optuna.csv \\")
    print(f"        -m \"WS2.5 multiseed-avg + RMSE-MH Optuna ensemble\"")
else:
    print(f"\n    Submit command (multiseed avg only):")
    print(f"      kaggle competitions submit -c m5-forecasting-accuracy \\")
    print(f"        -f submissions/lgbm_tvp_1p3_multiseed_avg.csv \\")
    print(f"        -m \"WS2.5 tvp=1.3 5-seed average\"")
print("\nDone!")
