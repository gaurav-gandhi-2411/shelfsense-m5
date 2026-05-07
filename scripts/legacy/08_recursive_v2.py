"""
Day 8: Recursive forecast v2 — exact day-index feature computation.

Stages:
  1. Load sales/calendar/prices + Day 6 global + Day 7 per-category models
  2. Single-step baseline (from parquet) — ground truth for sanity check
  3. v2 sanity check: recursive on d_1914-1941 with history through d_1913
  4. v2 eval forecast: d_1942-1969 (global + per-category + blend)
  5. Build three Kaggle submissions
  6. Print results for manual Kaggle submission
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
import lightgbm as lgb

from features.hierarchy import CAT_DTYPES
from models.recursive_forecast_v2 import (
    predict_horizon, _build_history_df,
    ALL_FEATURES, NUM_FEATURES, CAT_COLS,
)
from evaluation.wrmsse import compute_wrmsse

print("=" * 60)
print("DAY 8 -- Recursive Forecast v2")
print("=" * 60)

DATA_RAW = os.path.join(PROJ_ROOT, "data", "raw", "m5-forecasting-accuracy")
FEAT_DIR = os.path.join(PROJ_ROOT, "data", "processed", "features")
MODELS   = os.path.join(PROJ_ROOT, "data", "models")
REPORTS  = os.path.join(PROJ_ROOT, "reports")
SUBS     = os.path.join(PROJ_ROOT, "submissions")

LAST_TRAIN = 1913
HORIZON    = 28
VAL_START  = LAST_TRAIN + 1          # 1914
VAL_END    = LAST_TRAIN + HORIZON     # 1941
EVAL_START = VAL_END + 1             # 1942
EVAL_END   = VAL_END + HORIZON       # 1969
CATEGORIES = ["FOODS", "HOUSEHOLD", "HOBBIES"]

# ── 1. Load raw data ──────────────────────────────────────────────────────────
print("\n[1] Loading raw CSVs...")
t0 = time.time()
sales_eval  = pd.read_csv(os.path.join(DATA_RAW, "sales_train_evaluation.csv"))
prices_df   = pd.read_csv(os.path.join(DATA_RAW, "sell_prices.csv"))
calendar_df = pd.read_csv(os.path.join(DATA_RAW, "calendar.csv"))
print(f"    Loaded in {time.time()-t0:.1f}s")

# Load forecast feature parquet (d_1914-1941) for single-step baseline
print("\n    Loading forecast features parquet (d_1914-1941)...")
df_fcast = pd.read_parquet(
    FEAT_DIR,
    filters=[("d_num", ">=", VAL_START), ("d_num", "<=", VAL_END)],
    columns=["id", "item_id", "cat_id", "dept_id", "store_id", "state_id",
             "d_num", "sales"] + NUM_FEATURES,
)
for col, dtype in CAT_DTYPES.items():
    if col in df_fcast.columns:
        df_fcast[col] = df_fcast[col].astype(dtype)
df_fcast = df_fcast.sort_values(["id", "d_num"]).reset_index(drop=True)

series_meta = (
    df_fcast[["id", "item_id", "cat_id", "dept_id", "store_id", "state_id"]]
    .drop_duplicates("id").sort_values("id").reset_index(drop=True)
)
series_ids = series_meta["id"].values
n_series   = len(series_ids)
print(f"    Forecast parquet: {df_fcast.shape}  ({n_series:,} series)")

# ── 2. Load models ────────────────────────────────────────────────────────────
print("\n[2] Loading models...")
with open(os.path.join(MODELS, "lgbm_best.pkl"), "rb") as f:
    model_global = pickle.load(f)
print("    Loaded lgbm_best.pkl (Day 6 global)")

cat_models = {}
for cat in CATEGORIES:
    path = os.path.join(MODELS, f"lgbm_per_category_{cat}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            cat_models[cat] = pickle.load(f)
        print(f"    Loaded lgbm_per_category_{cat}.pkl")
    else:
        print(f"    WARNING: {path} not found — per-category blend will use global for {cat}")
        cat_models[cat] = model_global

# ── helpers ───────────────────────────────────────────────────────────────────
actual_val_cols = [f"d_{d}" for d in range(VAL_START, VAL_END + 1)]

def score_preds(preds_mat, sids, last_train=LAST_TRAIN):
    sub = sales_eval[sales_eval["id"].isin(sids)].set_index("id").reindex(sids).reset_index()
    acts = sub[actual_val_cols].values.astype(np.float32)
    s, _ = compute_wrmsse(
        preds=preds_mat, actuals=acts,
        sales_df=sub, prices_df=prices_df,
        calendar_df=calendar_df, last_train_day=last_train,
    )
    return float(s)

def single_step_preds(model):
    flat = np.clip(model.predict(df_fcast[ALL_FEATURES]), 0, None).astype(np.float32)
    return flat.reshape(n_series, HORIZON)

# ── 3. Single-step baseline ───────────────────────────────────────────────────
print("\n[3] Single-step baseline (parquet features, same as Day 6/7)...")
preds_ss = single_step_preds(model_global)
wrmsse_ss = score_preds(preds_ss, series_ids)
print(f"    Single-step WRMSSE: {wrmsse_ss:.4f}")

# ── 4. v2 sanity check: recursive on d_1914-1941 ─────────────────────────────
print(f"\n[4] v2 sanity check — recursive on d_{VAL_START}-{VAL_END}...")
print("    Building history DataFrame (d_1714-d_1913)...")
history_val = _build_history_df(sales_eval, series_ids, last_day=LAST_TRAIN)
print(f"    History: {len(history_val):,} rows  d_num range [{history_val['d_num'].min()}, {history_val['d_num'].max()}]")

t0 = time.time()
preds_v2_val, _ = predict_horizon(
    model_global, history_val, calendar_df, prices_df,
    days_out=HORIZON, cat_dtypes=CAT_DTYPES, verbose=True,
)
wrmsse_v2_val = score_preds(preds_v2_val, series_ids)
print(f"    v2 recursive WRMSSE:  {wrmsse_v2_val:.4f}")
print(f"    Single-step WRMSSE:   {wrmsse_ss:.4f}")
gap = wrmsse_v2_val - wrmsse_ss
print(f"    Gap: {gap:.4f} ({100*gap/wrmsse_ss:.1f}%)")
print(f"    [Expected: 8-12% gap is normal for 28-step recursive on sparse M5 data]")

# ── 5. v2 eval forecast: global model (d_1942-1969) ──────────────────────────
print(f"\n[5a] v2 eval forecast — global model (d_{EVAL_START}-{EVAL_END})...")
history_eval = _build_history_df(sales_eval, series_ids, last_day=VAL_END)
print(f"    History: d_num range [{history_eval['d_num'].min()}, {history_eval['d_num'].max()}]")

t0 = time.time()
eval_preds_global, _ = predict_horizon(
    model_global, history_eval, calendar_df, prices_df,
    days_out=HORIZON, cat_dtypes=CAT_DTYPES, verbose=True,
)
print(f"    Global eval done in {time.time()-t0:.1f}s")

# ── 5b. Per-category eval recursive ──────────────────────────────────────────
print(f"\n[5b] v2 eval forecast — per-category models...")
eval_preds_percat = np.zeros((n_series, HORIZON), dtype=np.float32)

for cat in CATEGORIES:
    cat_mask    = series_meta["cat_id"].astype(str) == cat
    cat_indices = np.where(cat_mask)[0]
    cat_sids    = series_ids[cat_indices]

    cat_history = history_eval[history_eval["id"].isin(cat_sids)]
    print(f"\n    {cat} ({len(cat_sids):,} series):")
    t0 = time.time()
    cat_preds, _ = predict_horizon(
        cat_models[cat], cat_history, calendar_df, prices_df,
        days_out=HORIZON, cat_dtypes=CAT_DTYPES, verbose=True,
    )
    eval_preds_percat[cat_indices] = cat_preds
    print(f"    {cat} done in {time.time()-t0:.1f}s")

# ── 5c. Blend ─────────────────────────────────────────────────────────────────
eval_preds_blend = 0.6 * eval_preds_percat + 0.4 * eval_preds_global

# ── 6. Validation predictions for submission header ──────────────────────────
# Val rows use single-step predictions (best accuracy; recursive not needed for val)
# Per-category val predictions
print("\n[6] Building per-category validation predictions (single-step)...")
val_preds_percat = np.zeros((n_series, HORIZON), dtype=np.float32)
for cat in CATEGORIES:
    cat_mask = df_fcast["cat_id"].astype(str) == cat
    cat_df   = df_fcast[cat_mask].sort_values(["id", "d_num"])
    cat_sids = cat_df.groupby("id", sort=True).first().reset_index()["id"].values
    flat     = np.clip(cat_models[cat].predict(cat_df[ALL_FEATURES]), 0, None).astype(np.float32)
    mat      = flat.reshape(len(cat_sids), HORIZON)
    idx      = np.searchsorted(series_ids, cat_sids)
    val_preds_percat[idx] = mat

val_preds_blend = 0.6 * val_preds_percat + 0.4 * preds_ss

wrmsse_percat_val = score_preds(val_preds_percat, series_ids)
wrmsse_blend_val  = score_preds(val_preds_blend, series_ids)
print(f"    Global single-step WRMSSE (val):    {wrmsse_ss:.4f}")
print(f"    Per-category WRMSSE (val):           {wrmsse_percat_val:.4f}")
print(f"    Blend 0.6/0.4 WRMSSE (val):         {wrmsse_blend_val:.4f}")

# ── 7. Build Kaggle submissions ───────────────────────────────────────────────
print("\n[7] Building Kaggle submissions...")
sn28_base = pd.read_csv(os.path.join(SUBS, "seasonal_naive_28_submission.csv"))
f_cols = [f"F{i}" for i in range(1, HORIZON + 1)]

def build_submission(val_mat, eval_mat, sids, fname):
    """
    Build 60,980-row submission:
      _validation rows (public LB)  → val_mat   (single-step, d_1914-1941)
      _evaluation rows (private LB) → eval_mat  (v2 recursive, d_1942-1969)
    """
    base = sn28_base.copy().set_index("id")

    val_ids = [s.replace("_evaluation", "_validation") for s in sids]
    val_df  = pd.DataFrame(val_mat, columns=f_cols)
    val_df.insert(0, "id", val_ids)
    val_idx = pd.Index(val_ids).intersection(base.index)
    base.loc[val_idx] = val_df.set_index("id").loc[val_idx]

    eval_df = pd.DataFrame(eval_mat, columns=f_cols)
    eval_df.insert(0, "id", list(sids))
    eval_idx = pd.Index(sids).intersection(base.index)
    base.loc[eval_idx] = eval_df.set_index("id").loc[eval_idx]

    path = os.path.join(SUBS, fname)
    base.reset_index().to_csv(path, index=False)
    print(f"    Saved: {fname}  ({len(val_idx)} val + {len(eval_idx)} eval rows)")
    return path

build_submission(preds_ss,         eval_preds_global, series_ids, "lgbm_v2_global.csv")
build_submission(val_preds_percat, eval_preds_percat, series_ids, "lgbm_v2_percat.csv")
build_submission(val_preds_blend,  eval_preds_blend,  series_ids, "lgbm_v2_blend.csv")

# ── 8. Save scores JSON ───────────────────────────────────────────────────────
results = {
    "day8_recursive_v2": {
        "sanity_check": {
            "single_step_wrmsse":    wrmsse_ss,
            "v2_recursive_wrmsse":   wrmsse_v2_val,
            "gap_absolute":          float(wrmsse_v2_val - wrmsse_ss),
            "gap_pct":               float(100 * (wrmsse_v2_val - wrmsse_ss) / wrmsse_ss),
        },
        "val_wrmsse": {
            "global_single_step":    wrmsse_ss,
            "percat_single_step":    wrmsse_percat_val,
            "blend_0.6_0.4":         wrmsse_blend_val,
        },
        "submissions": {
            "lgbm_v2_global":  "submit for public LB comparison",
            "lgbm_v2_percat":  "per-category val + recursive eval",
            "lgbm_v2_blend":   "RECOMMENDED — blend val + blend recursive eval",
        },
    }
}

def make_serial(obj):
    if isinstance(obj, (np.floating, float)): return float(obj)
    if isinstance(obj, (np.integer, int)):    return int(obj)
    if isinstance(obj, dict):  return {k: make_serial(v) for k, v in obj.items()}
    if isinstance(obj, list):  return [make_serial(x) for x in obj]
    return obj

scores_path = os.path.join(REPORTS, "day8_scores.json")
with open(scores_path, "w") as f:
    json.dump(make_serial(results), f, indent=2)
print(f"\n    Saved: {scores_path}")

# ── summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("DAY 8 SUMMARY")
print("=" * 60)
print(f"\nSanity check (v2 recursive vs single-step, val period):")
print(f"  Single-step WRMSSE:   {wrmsse_ss:.4f}")
print(f"  v2 recursive WRMSSE:  {wrmsse_v2_val:.4f}")
print(f"  Gap: {wrmsse_v2_val-wrmsse_ss:.4f} ({100*(wrmsse_v2_val-wrmsse_ss)/wrmsse_ss:.1f}%)")
print(f"\nValidation WRMSSE (single-step, 30490 series):")
print(f"  Global:       {wrmsse_ss:.4f}")
print(f"  Per-category: {wrmsse_percat_val:.4f}")
print(f"  Blend 0.6/0.4:{wrmsse_blend_val:.4f}")
print(f"\nSubmissions (val=single-step, eval=v2 recursive):")
print(f"  lgbm_v2_global.csv   — global model, recursive eval")
print(f"  lgbm_v2_percat.csv   — per-cat, recursive eval")
print(f"  lgbm_v2_blend.csv    — RECOMMENDED blend")
print(f"\nTo submit:")
print(f"  kaggle competitions submit -c m5-forecasting-accuracy -f submissions/lgbm_v2_blend.csv -m \"Day 8: v2 recursive blend\"")
print("\nDone!")
