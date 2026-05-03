"""
WS2.5 Variant 2 — Tweedie Variance Power Sweep

Trains 28 direct-horizon LightGBM models with a fixed tvp value.
Uses identical architecture/features/hyperparameters as Day 9 mh_blend,
varying only tweedie_variance_power:
  tvp=1.3  zero-inflation emphasis (compound Poisson, sparse/intermittent demand)
  tvp=1.5  mh_blend baseline (Optuna chose 1.53; 1.5 is a clean reference)
  tvp=1.7  spike emphasis (closer to gamma, high-volume concentration)

Usage:
  python scripts/12_lgbm_tvp_sweep.py 1.3
  python scripts/12_lgbm_tvp_sweep.py 1.5
  python scripts/12_lgbm_tvp_sweep.py 1.7

Generates (per tvp):
  data/models/tvp_{tag}/h_{h:02d}.pkl        — 28 trained models
  data/predictions/lgbm_tvp_{tag}_val.parquet
  data/predictions/lgbm_tvp_{tag}_eval.parquet
  submissions/lgbm_tvp_{tag}.csv
  reports/day12_tvp_{tag}_scores.json
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

from features.hierarchy import CAT_DTYPES
from evaluation.wrmsse import compute_wrmsse

# ── TVP argument ──────────────────────────────────────────────────────────────
if len(sys.argv) < 2:
    print("Usage: python scripts/12_lgbm_tvp_sweep.py <tvp>")
    print("  tvp: 1.3, 1.5, or 1.7")
    sys.exit(1)

TVP = float(sys.argv[1])
if TVP not in (1.3, 1.5, 1.7):
    print(f"Warning: tvp={TVP} is outside the planned sweep (1.3/1.5/1.7). Proceeding anyway.")

# Tag used in file/dir names (e.g. 1.3 -> "1p3")
TVP_TAG = f"{TVP:.1f}".replace(".", "p")

print("=" * 65)
print(f"WS2.5 Variant 2 — Tweedie Power Sweep  tvp={TVP}  (tag={TVP_TAG})")
print("=" * 65)

# ── paths ─────────────────────────────────────────────────────────────────────
DATA_RAW  = os.path.join(PROJ_ROOT, "data", "raw", "m5-forecasting-accuracy")
FEAT_DIR  = os.path.join(PROJ_ROOT, "data", "processed", "features")
MODEL_DIR = os.path.join(PROJ_ROOT, "data", "models", f"tvp_{TVP_TAG}")
PREDS_DIR = os.path.join(PROJ_ROOT, "data", "predictions")
REPORTS   = os.path.join(PROJ_ROOT, "reports")
SUBS      = os.path.join(PROJ_ROOT, "submissions")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PREDS_DIR, exist_ok=True)

# ── constants (identical to Day 9) ────────────────────────────────────────────
LAST_TRAIN   = 1913
VAL_START    = 1886
FEAT_START   = 1000
HORIZON      = 28

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

# Day 9 Optuna best params with tvp overridden
BEST_PARAMS = {
    "objective":              "tweedie",
    "metric":                 "tweedie",
    "verbose":               -1,
    "num_threads":            0,
    "seed":                   42,
    "bagging_freq":           1,
    "learning_rate":          0.025,
    "num_leaves":             64,
    "min_data_in_leaf":       100,
    "feature_fraction":       0.7,
    "bagging_fraction":       0.9,
    "lambda_l2":              0.1,
    "tweedie_variance_power": TVP,   # only thing that changes
}

print(f"\nFixed params (Day 9 Optuna best, tvp overridden):")
print(f"  lr={BEST_PARAMS['learning_rate']}  leaves={BEST_PARAMS['num_leaves']}  "
      f"min_data={BEST_PARAMS['min_data_in_leaf']}  l2={BEST_PARAMS['lambda_l2']}")
print(f"  feat_frac={BEST_PARAMS['feature_fraction']}  bag_frac={BEST_PARAMS['bagging_fraction']}")
print(f"  tvp={TVP}  (Day 9 Optuna chose 1.5316)")

# ── 1. Load training data ─────────────────────────────────────────────────────
print(f"\n[1] Loading training data (d_num {FEAT_START}–{LAST_TRAIN})...")
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
df = df.dropna(subset=["lag_7", "lag_14", "lag_28", "lag_56"]).reset_index(drop=True)
print(f"    Rows: {len(df):,}  ({time.time()-t0:.1f}s)")

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
print(f"    Series: {n_series:,}")

actual_val_cols = [f"d_{LAST_TRAIN + h}" for h in range(1, HORIZON + 1)]
f_cols = [f"F{i}" for i in range(1, HORIZON + 1)]

def score_preds(preds_mat, sids):
    sub  = sales_eval[sales_eval["id"].isin(sids)].set_index("id").reindex(sids).reset_index()
    acts = sub[actual_val_cols].values.astype(np.float32)
    s, _ = compute_wrmsse(
        preds=preds_mat, actuals=acts,
        sales_df=sub, prices_df=prices_df,
        calendar_df=calendar_df, last_train_day=LAST_TRAIN,
    )
    return float(s)

# ── 3. Train 28 horizon models ────────────────────────────────────────────────
print(f"\n[3] Training 28 models (tvp={TVP})...")
models   = {}
h_scores = {}
t_total  = time.time()

for h in range(1, HORIZON + 1):
    path = os.path.join(MODEL_DIR, f"h_{h:02d}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            models[h] = pickle.load(f)
        print(f"  h={h:2d}: loaded from cache")
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
        BEST_PARAMS, ds_tr,
        num_boost_round=3000,
        valid_sets=[ds_vl],
        callbacks=[lgb.early_stopping(75, verbose=False), lgb.log_evaluation(500)],
    )
    elapsed = time.time() - t0

    with open(path, "wb") as f:
        pickle.dump(model, f)

    val_metric = float(model.best_score["valid_0"]["tweedie"])
    h_scores[h] = {
        "best_iter":   model.best_iteration,
        "val_tweedie": val_metric,
        "train_s":     float(elapsed),
    }
    print(f"  h={h:2d}: iter={model.best_iteration:4d}  val={val_metric:.4f}  {elapsed:.0f}s  saved")
    models[h] = model

    del ds_tr, ds_vl, X_tr, y_tr, X_vl, y_vl, y_h
    gc.collect()

total_train_min = (time.time() - t_total) / 60
print(f"\n    All 28 models in {total_train_min:.1f} min")

# ── 4. Val WRMSSE (multi-horizon from d_1913) ─────────────────────────────────
print(f"\n[4] Val WRMSSE (multi-horizon from d_{LAST_TRAIN})...")
val_preds = np.zeros((n_series, HORIZON), dtype=np.float32)
for h in range(1, HORIZON + 1):
    val_preds[:, h-1] = np.clip(models[h].predict(df_val_origin[ALL_FEATURES]), 0.0, None)

val_wrmsse = score_preds(val_preds, series_ids)
print(f"    tvp={TVP} val WRMSSE:   {val_wrmsse:.4f}")
print(f"    Day 9 reference:       0.7254  (mh_blend tvp=1.5316)")
print(f"    Delta:                 {val_wrmsse - 0.7254:+.4f}")

# ── 5. Eval predictions from d_1941 ──────────────────────────────────────────
print(f"\n[5] Eval predictions (from d_{LAST_TRAIN + HORIZON})...")
eval_preds = np.zeros((n_series, HORIZON), dtype=np.float32)
for h in range(1, HORIZON + 1):
    eval_preds[:, h-1] = np.clip(models[h].predict(df_eval_origin[ALL_FEATURES]), 0.0, None)

# ── 6. Save prediction parquets ──────────────────────────────────────────────
print(f"\n[6] Saving prediction parquets...")
val_df = pd.DataFrame(val_preds, columns=f_cols)
val_df.insert(0, "id", series_ids)
val_df.to_parquet(os.path.join(PREDS_DIR, f"lgbm_tvp_{TVP_TAG}_val.parquet"), index=False)

eval_df = pd.DataFrame(eval_preds, columns=f_cols)
eval_df.insert(0, "id", series_ids)
eval_df.to_parquet(os.path.join(PREDS_DIR, f"lgbm_tvp_{TVP_TAG}_eval.parquet"), index=False)
print(f"    Saved lgbm_tvp_{TVP_TAG}_val.parquet and lgbm_tvp_{TVP_TAG}_eval.parquet")

# ── 7. Load single-step val preds for submission val rows ─────────────────────
print(f"\n[7] Loading single-step preds for submission val rows...")
df_ss = pd.read_parquet(
    FEAT_DIR,
    filters=[("d_num", ">=", LAST_TRAIN + 1), ("d_num", "<=", LAST_TRAIN + HORIZON)],
    columns=["id", "d_num"] + ALL_FEATURES,
)
for col, dtype in CAT_DTYPES.items():
    if col in df_ss.columns:
        df_ss[col] = df_ss[col].astype(dtype)
df_ss = df_ss.sort_values(["id", "d_num"]).reset_index(drop=True)

with open(os.path.join(PROJ_ROOT, "data", "models", "lgbm_best.pkl"), "rb") as _f:
    model_global = pickle.load(_f)

val_preds_ss = np.clip(
    model_global.predict(df_ss[ALL_FEATURES]), 0, None
).astype(np.float32).reshape(n_series, HORIZON)
print(f"    Single-step val rows loaded ({len(df_ss):,} rows)")

# ── 8. Build Kaggle submission ────────────────────────────────────────────────
print(f"\n[8] Building submission (lgbm_tvp_{TVP_TAG}.csv)...")
sn28 = pd.read_csv(os.path.join(SUBS, "seasonal_naive_28_submission.csv"))

def build_submission(val_mat, eval_mat, sids, fname):
    base     = sn28.copy().set_index("id")
    val_ids  = [s.replace("_evaluation", "_validation") for s in sids]
    vdf      = pd.DataFrame(val_mat, columns=f_cols)
    vdf.insert(0, "id", val_ids)
    val_idx  = pd.Index(val_ids).intersection(base.index)
    base.loc[val_idx] = vdf.set_index("id").loc[val_idx]
    edf      = pd.DataFrame(eval_mat, columns=f_cols)
    edf.insert(0, "id", list(sids))
    eval_idx = pd.Index(sids).intersection(base.index)
    base.loc[eval_idx] = edf.set_index("id").loc[eval_idx]
    path = os.path.join(SUBS, fname)
    base.reset_index().to_csv(path, index=False)
    print(f"    Saved {fname}  ({len(val_idx)} val + {len(eval_idx)} eval rows)")

build_submission(val_preds_ss, eval_preds, series_ids, f"lgbm_tvp_{TVP_TAG}.csv")

# ── 9. Save results JSON ──────────────────────────────────────────────────────
def make_serial(obj):
    if isinstance(obj, (np.floating, float)): return float(obj)
    if isinstance(obj, (np.integer, int)):    return int(obj)
    if isinstance(obj, dict):  return {k: make_serial(v) for k, v in obj.items()}
    if isinstance(obj, list):  return [make_serial(x) for x in obj]
    return obj

results = {
    "variant":           "lgbm_tvp_sweep",
    "tvp":               TVP,
    "tvp_tag":           TVP_TAG,
    "params":            {k: float(v) if isinstance(v, float) else v for k, v in BEST_PARAMS.items()},
    "total_train_min":   total_train_min,
    "val_wrmsse":        val_wrmsse,
    "reference_val_wrmsse_day9": 0.7254,
    "reference_private_lb_day9": 0.5854,
    "per_horizon": {str(h): h_scores.get(h, {}) for h in range(1, HORIZON + 1)},
}

scores_path = os.path.join(REPORTS, f"day12_tvp_{TVP_TAG}_scores.json")
with open(scores_path, "w") as f:
    json.dump(make_serial(results), f, indent=2)
print(f"\n    Saved {scores_path}")

# ── summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print(f"WS2.5 VARIANT 2 (tvp={TVP}) — SUMMARY")
print("=" * 65)
print(f"\nTotal training time: {total_train_min:.1f} min  (28 models)")
print(f"Val WRMSSE:          {val_wrmsse:.4f}  (Day 9 ref: 0.7254)")
print(f"Delta:               {val_wrmsse - 0.7254:+.4f}")
print(f"\nSubmission: submissions/lgbm_tvp_{TVP_TAG}.csv")
print(f"\nTo submit to Kaggle:")
print(f'  kaggle competitions submit -c m5-forecasting-accuracy \\')
print(f'    -f submissions/lgbm_tvp_{TVP_TAG}.csv \\')
print(f'    -m "WS2.5 Variant 2: Tweedie power sweep tvp={TVP}"')
print(f"\nNext tvp values: {[v for v in [1.3, 1.5, 1.7] if v != TVP]}")
print(f"  python scripts/12_lgbm_tvp_sweep.py {[v for v in [1.3, 1.5, 1.7] if v != TVP][0]}")
print("\nDone!")
