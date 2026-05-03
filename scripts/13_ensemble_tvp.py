"""
WS2.5 — Tweedie Power Sweep Ensemble Experiments

Builds and submits 4 ensemble candidates from available LightGBM variants:
  mh_blend  (Day 9, tvp=1.5316, private LB 0.5854)
  tvp=1.3   (WS2.5 V2a, private LB 0.5693 — new best)
  tvp=1.7   (WS2.5 V2c, private LB 0.6623)

All val WRMSSE computed from the SAME origin (d_1913) using multi-horizon
direct predictions. Day 9 pkl models are reloaded to get comparable mh_blend
val preds — avoids oracle bias from extracting single-step rows from the
submission CSV.

Ensemble candidates (eval rows only; val rows always use single-step oracle):
  1. tvp13_mhblend_5050     — 0.5 × tvp=1.3  + 0.5 × mh_blend
  2. tvp13_tvp17_5050       — 0.5 × tvp=1.3  + 0.5 × tvp=1.7
  3. tvp13_tvp17_mh_3333    — 1/3 × each of tvp=1.3, tvp=1.7, mh_blend
  4. optuna_blend           — Optuna-optimized weights across all 3

Usage:
  python scripts/13_ensemble_tvp.py
"""
from __future__ import annotations
import sys, os, time, json, pickle, warnings
warnings.filterwarnings("ignore")

_SP = "C:/Users/gaura/anaconda3/Lib/site-packages"
if _SP not in sys.path:
    sys.path.insert(0, _SP)

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_ROOT)
sys.path.insert(0, os.path.join(PROJ_ROOT, "src"))

import numpy as np
import pandas as pd
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from features.hierarchy import CAT_DTYPES
from evaluation.wrmsse import compute_wrmsse

print("=" * 65)
print("WS2.5 — Tweedie Power Sweep Ensemble Experiments")
print("=" * 65)

# ── paths ─────────────────────────────────────────────────────────────────────
DATA_RAW  = os.path.join(PROJ_ROOT, "data", "raw", "m5-forecasting-accuracy")
FEAT_DIR  = os.path.join(PROJ_ROOT, "data", "processed", "features")
PREDS_DIR = os.path.join(PROJ_ROOT, "data", "predictions")
REPORTS   = os.path.join(PROJ_ROOT, "reports")
SUBS      = os.path.join(PROJ_ROOT, "submissions")
MODELS    = os.path.join(PROJ_ROOT, "data", "models")
MH_DIR    = os.path.join(MODELS, "multi_horizon")

# Must match Day 9 ALL_FEATURES exactly (model was trained on these)
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

LAST_TRAIN = 1913
HORIZON    = 28
f_cols     = [f"F{i}" for i in range(1, HORIZON + 1)]
actual_val_cols = [f"d_{LAST_TRAIN + h}" for h in range(1, HORIZON + 1)]

# ── 1. Load raw files for scoring ─────────────────────────────────────────────
print("\n[1] Loading raw CSVs...")
t0 = time.time()
sales_eval  = pd.read_csv(os.path.join(DATA_RAW, "sales_train_evaluation.csv"))
prices_df   = pd.read_csv(os.path.join(DATA_RAW, "sell_prices.csv"))
calendar_df = pd.read_csv(os.path.join(DATA_RAW, "calendar.csv"))
print(f"    Loaded in {time.time()-t0:.1f}s")

# ── 2. Load all variant predictions ──────────────────────────────────────────
print("\n[2] Loading variant predictions...")

def load_preds(fname) -> np.ndarray:
    df = pd.read_parquet(os.path.join(PREDS_DIR, fname))
    series_ids = df["id"].values
    return series_ids, df[f_cols].values.astype(np.float32)

print("    Loading mh_blend (Day 9 submission eval rows)...")
# mh_blend eval rows come from the Day 9 submission file, not a parquet
# Reconstruct from mh_global eval rows (the pure MH preds before blending with Day8)
# Actually mh_blend = 0.5 * mh_global + 0.5 * day8_blend (from Day 9 script step 8)
# We need the pure tvp=1.5316 eval preds. These were not saved as a separate parquet.
# Reconstruct: load mh_blend submission, subtract day8 eval component to get mh_global,
# then use that as the reference.
# Simpler: use the mh_blend submission directly as "mh_blend" variant for ensembling.
sn28     = pd.read_csv(os.path.join(SUBS, "seasonal_naive_28_submission.csv"))
mh_sub   = pd.read_csv(os.path.join(SUBS, "mh_blend.csv")).set_index("id")

series_ids_1p3, eval_1p3 = load_preds("lgbm_tvp_1p3_eval.parquet")
series_ids_1p7, eval_1p7 = load_preds("lgbm_tvp_1p7_eval.parquet")

# Verify series alignment
assert np.array_equal(series_ids_1p3, series_ids_1p7), "series_ids mismatch between tvp=1.3 and tvp=1.7"
series_ids = series_ids_1p3
n_series   = len(series_ids)
print(f"    tvp=1.3 preds: {eval_1p3.shape}")
print(f"    tvp=1.7 preds: {eval_1p7.shape}")

# Extract mh_blend eval rows (evaluation suffix IDs)
mh_eval_mat = mh_sub.loc[series_ids, f_cols].values.astype(np.float32)
print(f"    mh_blend eval rows: {mh_eval_mat.shape}")

# ── 3. Load val origin for WRMSSE scoring ─────────────────────────────────────
print("\n[3] Loading val origin for WRMSSE scoring...")
df_val_origin = pd.read_parquet(
    FEAT_DIR,
    filters=[("d_num", "==", LAST_TRAIN)],
    columns=["id", "cat_id", "dept_id", "store_id", "state_id"] +
            ["weekday", "month", "quarter", "year", "day_of_month", "week_of_year",
             "is_weekend", "is_holiday", "is_snap_ca", "is_snap_tx", "is_snap_wi",
             "days_since_event", "days_until_next_event",
             "sell_price", "price_change_pct", "price_relative_mean",
             "price_volatility", "has_price_change",
             "lag_7", "lag_14", "lag_28", "lag_56",
             "roll_mean_7", "roll_std_7", "roll_min_7", "roll_max_7",
             "roll_mean_28", "roll_std_28", "roll_min_28", "roll_max_28",
             "roll_mean_56", "roll_std_56", "roll_min_56", "roll_max_56",
             "roll_mean_180","roll_std_180","roll_min_180","roll_max_180"],
)
for col, dt in {c: CAT_DTYPES[c] for c in CAT_DTYPES if c in df_val_origin.columns}.items():
    df_val_origin[col] = df_val_origin[col].astype(dt)
df_val_origin = df_val_origin.sort_values("id").reset_index(drop=True)

# Load val predictions for each variant
val_1p3 = pd.read_parquet(os.path.join(PREDS_DIR, "lgbm_tvp_1p3_val.parquet"))
val_1p3 = val_1p3.set_index("id").loc[series_ids, f_cols].values.astype(np.float32)

val_1p7 = pd.read_parquet(os.path.join(PREDS_DIR, "lgbm_tvp_1p7_val.parquet"))
val_1p7 = val_1p7.set_index("id").loc[series_ids, f_cols].values.astype(np.float32)

# Same-origin mh_blend val predictions: reload Day 9 pkl models, predict from d_1913.
# This avoids oracle bias from the submission CSV's single-step rows.
print("    Loading Day 9 multi-horizon models for same-origin val preds...")
t0 = time.time()
mh_models = {}
for h in range(1, HORIZON + 1):
    with open(os.path.join(MH_DIR, f"h_{h:02d}.pkl"), "rb") as _f:
        mh_models[h] = pickle.load(_f)
val_mh = np.zeros((n_series, HORIZON), dtype=np.float32)
for h in range(1, HORIZON + 1):
    val_mh[:, h-1] = np.clip(
        mh_models[h].predict(df_val_origin[ALL_FEATURES]), 0.0, None
    ).astype(np.float32)
del mh_models
print(f"    Day 9 val preds done in {time.time()-t0:.1f}s  shape={val_mh.shape}")
val_ids = [s.replace("_evaluation", "_validation") for s in series_ids]
# Oracle single-step val rows for submission (lgbm_best.pkl preds via mh_blend CSV).
# Used ONLY in the submission file — not for Optuna/WRMSSE scoring.
val_oracle_mat = mh_sub.loc[val_ids, f_cols].values.astype(np.float32)

def score_preds(preds_mat, sids):
    sub  = sales_eval[sales_eval["id"].isin(sids)].set_index("id").reindex(sids).reset_index()
    acts = sub[actual_val_cols].values.astype(np.float32)
    s, _ = compute_wrmsse(
        preds=preds_mat, actuals=acts,
        sales_df=sub, prices_df=prices_df,
        calendar_df=calendar_df, last_train_day=LAST_TRAIN,
    )
    return float(s)

print("\n[3b] Individual val WRMSSE:")
w_1p3 = score_preds(val_1p3, series_ids)
w_1p7 = score_preds(val_1p7, series_ids)
w_mh  = score_preds(val_mh,  series_ids)
print(f"    tvp=1.3:  {w_1p3:.4f}")
print(f"    tvp=1.7:  {w_1p7:.4f}")
print(f"    mh_blend: {w_mh:.4f}  (same-origin d_1913 multi-horizon, comparable to tvp variants)")

# ── 4. Build and score ensemble candidates ────────────────────────────────────
print("\n[4] Building ensemble candidates (val WRMSSE for Optuna, eval preds for submission)...")

candidates = {
    "tvp13_mhblend_5050":  (0.5 * eval_1p3 + 0.5 * mh_eval_mat,
                             0.5 * val_1p3  + 0.5 * val_mh),
    "tvp13_tvp17_5050":    (0.5 * eval_1p3 + 0.5 * eval_1p7,
                             0.5 * val_1p3  + 0.5 * val_1p7),
    "tvp13_tvp17_mh_3333": ((eval_1p3 + eval_1p7 + mh_eval_mat) / 3.0,
                             (val_1p3  + val_1p7  + val_mh) / 3.0),
}

print("\n    Candidate val WRMSSEs (all same-origin d_1913, no oracle bias):")
for name, (_, vmat) in candidates.items():
    w = score_preds(vmat, series_ids)
    print(f"    {name}: {w:.4f}")

# ── 5. Optuna weight optimization ─────────────────────────────────────────────
print("\n[5] Optuna weight optimization (50 trials, val WRMSSE)...")

def optuna_objective(trial):
    w1 = trial.suggest_float("w_13",  0.0, 1.0)
    w2 = trial.suggest_float("w_17",  0.0, 1.0)
    w3 = trial.suggest_float("w_mh",  0.0, 1.0)
    total = w1 + w2 + w3
    if total < 1e-6:
        return 9999.0
    blend_val = (w1 * val_1p3 + w2 * val_1p7 + w3 * val_mh) / total
    return score_preds(blend_val, series_ids)

study = optuna.create_study(direction="minimize")
t0 = time.time()
study.optimize(optuna_objective, n_trials=50, show_progress_bar=False)
opt_time = time.time() - t0

bp = study.best_params
total_w = bp["w_13"] + bp["w_17"] + bp["w_mh"]
w_13_n  = bp["w_13"] / total_w
w_17_n  = bp["w_17"] / total_w
w_mh_n  = bp["w_mh"] / total_w

print(f"    Optuna done in {opt_time:.0f}s  best_val={study.best_value:.4f}")
print(f"    Optimal weights: tvp=1.3: {w_13_n:.3f}  tvp=1.7: {w_17_n:.3f}  mh_blend: {w_mh_n:.3f}")

optuna_eval = w_13_n * eval_1p3 + w_17_n * eval_1p7 + w_mh_n * mh_eval_mat
optuna_val  = w_13_n * val_1p3  + w_17_n * val_1p7  + w_mh_n * val_mh
candidates["optuna_blend"] = (optuna_eval, optuna_val)

# ── 6. Build submissions ───────────────────────────────────────────────────────
print("\n[6] Building submissions...")

def build_submission(eval_mat, fname):
    base     = sn28.copy().set_index("id")
    # val rows: use single-step oracle (lgbm_best.pkl preds, via mh_blend submission)
    vdf = pd.DataFrame(val_oracle_mat, columns=f_cols)
    vdf.insert(0, "id", val_ids)
    val_idx = pd.Index(val_ids).intersection(base.index)
    base.loc[val_idx] = vdf.set_index("id").loc[val_idx]
    # eval rows: ensemble
    edf = pd.DataFrame(eval_mat, columns=f_cols)
    edf.insert(0, "id", list(series_ids))
    eval_idx = pd.Index(series_ids).intersection(base.index)
    base.loc[eval_idx] = edf.set_index("id").loc[eval_idx]
    path = os.path.join(SUBS, fname)
    base.reset_index().to_csv(path, index=False)
    print(f"    Saved {fname}  ({len(val_idx)} val + {len(eval_idx)} eval rows)")

for name, (emat, _) in candidates.items():
    build_submission(np.clip(emat, 0.0, None).astype(np.float32), f"ens_{name}.csv")

# ── 7. Save results JSON ──────────────────────────────────────────────────────
def make_serial(obj):
    if isinstance(obj, (np.floating, float)): return float(obj)
    if isinstance(obj, (np.integer, int)):    return int(obj)
    if isinstance(obj, dict):  return {k: make_serial(v) for k, v in obj.items()}
    if isinstance(obj, list):  return [make_serial(x) for x in obj]
    return obj

results = {
    "individual_val_wrmsse": {"tvp_1p3": w_1p3, "tvp_1p7": w_1p7, "mh_blend": w_mh},
    "optuna": {
        "n_trials": 50, "best_val": study.best_value,
        "weights": {"w_13": w_13_n, "w_17": w_17_n, "w_mh": w_mh_n},
    },
    "candidate_val_wrmsse": {},
}
for name, (_, vmat) in candidates.items():
    results["candidate_val_wrmsse"][name] = score_preds(vmat, series_ids)

with open(os.path.join(REPORTS, "day13_ensemble_scores.json"), "w") as f:
    json.dump(make_serial(results), f, indent=2)
print(f"\n    Saved reports/day13_ensemble_scores.json")

# ── summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("ENSEMBLE SUMMARY")
print("=" * 65)
print(f"\nIndividual val WRMSSEs:")
print(f"  tvp=1.3:  {w_1p3:.4f}")
print(f"  tvp=1.7:  {w_1p7:.4f}")
print(f"  mh_blend: {w_mh:.4f}")
print(f"\nEnsemble candidate val WRMSSEs:")
for name, (_, vmat) in candidates.items():
    w = results["candidate_val_wrmsse"][name]
    print(f"  {name}: {w:.4f}")
print(f"\nOptuna best: {study.best_value:.4f}")
print(f"  Weights: tvp=1.3={w_13_n:.3f}  tvp=1.7={w_17_n:.3f}  mh_blend={w_mh_n:.3f}")
print(f"\nSubmissions built (submit each separately):")
for name in candidates:
    print(f"  submissions/ens_{name}.csv")
print(f"\nSubmit commands:")
for name in candidates:
    desc = name.replace("_", " ")
    print(f'  kaggle competitions submit -c m5-forecasting-accuracy -f submissions/ens_{name}.csv -m "WS2.5 ensemble: {desc}"')
print("\nDone!")
