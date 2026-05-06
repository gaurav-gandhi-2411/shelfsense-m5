"""
Stage A sanity train: yearly lag features (lag_91, lag_182, lag_364)

Trains a single global Tweedie model using Day 9 Optuna best params with the
new 41-feature set (+3 yearly lags). Reports:
  - Training time
  - Oracle val WRMSSE (predict d_1914-1941 using actual per-day features)
  - Full feature importance table (41 features ranked by gain)
  - GO / NO-GO signal: do lag_91/182/364 rank in top 20?

Baseline for comparison: lgbm_best.pkl oracle val WRMSSE = 0.5422

Does NOT run Optuna (Stage A is a sanity check, not a full experiment).
Stage B (28 multi-horizon models) proceeds only if yearly lags rank in top 20.

Usage:
  python scripts/15_sanity_train_lags.py
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
from evaluation.wrmsse import compute_wrmsse

print("=" * 65)
print("Stage A — Yearly Lag Sanity Train")
print("=" * 65)

DATA_RAW  = os.path.join(PROJ_ROOT, "data", "raw", "m5-forecasting-accuracy")
FEAT_DIR  = os.path.join(PROJ_ROOT, "data", "processed", "features")
REPORTS   = os.path.join(PROJ_ROOT, "reports")
MODELS    = os.path.join(PROJ_ROOT, "data", "models")

LAST_TRAIN  = 1913
VAL_START   = 1886
FEAT_START  = 1000
HORIZON     = 28
actual_val_cols = [f"d_{LAST_TRAIN + h}" for h in range(1, HORIZON + 1)]

# Day 9 Optuna best params — unchanged, only feature set grows
PARAMS = {
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
    "tweedie_variance_power": 1.5316,
}

CAT_FEATURES = ["cat_id", "dept_id", "store_id", "state_id"]
NUM_FEATURES = [
    "weekday", "month", "quarter", "year", "day_of_month", "week_of_year",
    "is_weekend", "is_holiday", "is_snap_ca", "is_snap_tx", "is_snap_wi",
    "days_since_event", "days_until_next_event",
    "sell_price", "price_change_pct", "price_relative_mean",
    "price_volatility", "has_price_change",
    "lag_7", "lag_14", "lag_28", "lag_56",
    "lag_91", "lag_182", "lag_364",            # new yearly lags
    "roll_mean_7",  "roll_std_7",  "roll_min_7",  "roll_max_7",
    "roll_mean_28", "roll_std_28", "roll_min_28", "roll_max_28",
    "roll_mean_56", "roll_std_56", "roll_min_56", "roll_max_56",
    "roll_mean_180","roll_std_180","roll_min_180","roll_max_180",
]
ALL_FEATURES = NUM_FEATURES + CAT_FEATURES
print(f"\n  Feature count: {len(ALL_FEATURES)} ({len(NUM_FEATURES)} numeric + {len(CAT_FEATURES)} categorical)")
print(f"  New features: lag_91, lag_182, lag_364")

# ── 1. Load training data ──────────────────────────────────────────────────────
print(f"\n[1] Loading training data (d_num {FEAT_START}-{LAST_TRAIN})...")
t0 = time.time()
df = pd.read_parquet(
    FEAT_DIR,
    filters=[("d_num", ">=", FEAT_START), ("d_num", "<=", LAST_TRAIN)],
    columns=["id", "cat_id", "dept_id", "store_id", "state_id",
             "d_num", "sales"] + NUM_FEATURES,
)
for col, dt in CAT_DTYPES.items():
    if col in df.columns:
        df[col] = df[col].astype(dt)
# Drop rows where any short lag is NaN (early series rows); lag_364 is always populated at d_num>=1000
df = df.dropna(subset=["lag_7", "lag_14", "lag_28", "lag_56"]).reset_index(drop=True)
print(f"    Rows: {len(df):,}  ({time.time()-t0:.1f}s)")

# ── 1b. Load raw CSVs for scoring ─────────────────────────────────────────────
print("\n[1b] Loading raw CSVs...")
t0 = time.time()
sales_eval  = pd.read_csv(os.path.join(DATA_RAW, "sales_train_evaluation.csv"))
prices_df   = pd.read_csv(os.path.join(DATA_RAW, "sell_prices.csv"))
calendar_df = pd.read_csv(os.path.join(DATA_RAW, "calendar.csv"))
print(f"     Loaded in {time.time()-t0:.1f}s")

# ── 2. Build LightGBM datasets ────────────────────────────────────────────────
print(f"\n[2] Building train/val datasets...")
train_mask = df["d_num"] <= VAL_START - 1
val_mask   = (df["d_num"] >= VAL_START) & (df["d_num"] <= LAST_TRAIN)

X_tr = df.loc[train_mask, ALL_FEATURES]
y_tr = df.loc[train_mask, "sales"].astype(np.float32)
X_vl = df.loc[val_mask,   ALL_FEATURES]
y_vl = df.loc[val_mask,   "sales"].astype(np.float32)
print(f"    Train: {len(X_tr):,}  Val: {len(X_vl):,}")

ds_tr = lgb.Dataset(X_tr, label=y_tr.values, categorical_feature=CAT_FEATURES, free_raw_data=False)
ds_vl = lgb.Dataset(X_vl, label=y_vl.values, categorical_feature=CAT_FEATURES, reference=ds_tr, free_raw_data=False)

# ── 3. Train ───────────────────────────────────────────────────────────────────
print(f"\n[3] Training single Tweedie model (Day 9 params, 41 features)...")
t0 = time.time()
model = lgb.train(
    PARAMS, ds_tr,
    num_boost_round=3000,
    valid_sets=[ds_vl],
    callbacks=[lgb.early_stopping(75, verbose=False), lgb.log_evaluation(500)],
)
train_time = time.time() - t0
print(f"    iter={model.best_iteration}  val_tweedie={model.best_score['valid_0']['tweedie']:.4f}")
print(f"    Training time: {train_time:.0f}s  ({train_time/60:.1f} min)")

# ── 4. Oracle val WRMSSE ──────────────────────────────────────────────────────
print(f"\n[4] Oracle val WRMSSE (predict d_{LAST_TRAIN+1}-{LAST_TRAIN+HORIZON} using actual features)...")
df_oracle = pd.read_parquet(
    FEAT_DIR,
    filters=[("d_num", ">=", LAST_TRAIN + 1), ("d_num", "<=", LAST_TRAIN + HORIZON)],
    columns=["id", "cat_id", "dept_id", "store_id", "state_id", "d_num"] + NUM_FEATURES,
)
for col, dt in CAT_DTYPES.items():
    if col in df_oracle.columns:
        df_oracle[col] = df_oracle[col].astype(dt)
df_oracle = df_oracle.sort_values(["id", "d_num"]).reset_index(drop=True)

preds_flat = np.clip(model.predict(df_oracle[ALL_FEATURES]), 0.0, None).astype(np.float32)
series_ids = df_oracle.groupby("id").first().reset_index()["id"].values
n_series   = len(series_ids)
preds_mat  = preds_flat.reshape(n_series, HORIZON)

sales_sub = sales_eval[sales_eval["id"].isin(series_ids)].set_index("id").reindex(series_ids).reset_index()
actuals   = sales_sub[actual_val_cols].values.astype(np.float32)
wrmsse, _ = compute_wrmsse(
    preds=preds_mat, actuals=actuals,
    sales_df=sales_sub, prices_df=prices_df,
    calendar_df=calendar_df, last_train_day=LAST_TRAIN,
)
wrmsse = float(wrmsse)
print(f"    Oracle val WRMSSE: {wrmsse:.4f}")
print(f"    lgbm_best baseline: 0.5422  (Day 9 params, 38 features)")
print(f"    Delta:              {wrmsse - 0.5422:+.4f}")

# ── 5. Feature importance ─────────────────────────────────────────────────────
print(f"\n[5] Feature importance (gain, all {len(ALL_FEATURES)} features)...")
importance = pd.DataFrame({
    "feature": model.feature_name(),
    "gain":    model.feature_importance(importance_type="gain"),
    "split":   model.feature_importance(importance_type="split"),
}).sort_values("gain", ascending=False).reset_index(drop=True)
importance["rank"] = importance.index + 1

print(f"\n    Full ranking by gain:")
print(f"    {'Rank':>4}  {'Feature':<22}  {'Gain':>12}  {'Split':>8}")
print(f"    {'-'*4}  {'-'*22}  {'-'*12}  {'-'*8}")
for _, row in importance.iterrows():
    marker = " ***" if row["feature"] in ("lag_91", "lag_182", "lag_364") else ""
    print(f"    {int(row['rank']):>4}  {row['feature']:<22}  {row['gain']:>12.0f}  {int(row['split']):>8}{marker}")

yearly_lags = importance[importance["feature"].isin(["lag_91", "lag_182", "lag_364"])]
print(f"\n    Yearly lag ranks:")
for _, row in yearly_lags.iterrows():
    print(f"      {row['feature']}: rank {int(row['rank'])}/{len(ALL_FEATURES)}  gain={row['gain']:.0f}")

top20_features = set(importance.head(20)["feature"].tolist())
yearly_in_top20 = [f for f in ["lag_91", "lag_182", "lag_364"] if f in top20_features]
yearly_all_below30 = all(int(r) >= 30 for r in yearly_lags["rank"])

# ── 6. GO / NO-GO ─────────────────────────────────────────────────────────────
print(f"\n[6] Stage B GO / NO-GO assessment...")
print(f"    Yearly lags in top 20: {yearly_in_top20 if yearly_in_top20 else 'none'}")
print(f"    All yearly lags below rank 30: {yearly_all_below30}")

if yearly_in_top20:
    verdict = "GO"
    reason  = f"{len(yearly_in_top20)}/3 yearly lags rank in top 20 — meaningful signal, Stage B justified"
elif not yearly_all_below30:
    verdict = "MARGINAL"
    reason  = "Some yearly lags rank 21-29 — review before deciding on Stage B"
else:
    verdict = "NO-GO"
    reason  = "All yearly lags rank ≥30/41 — yearly seasonality not contributing, Stage B not justified"

print(f"\n    VERDICT: {verdict}")
print(f"    Reason:  {reason}")

# ── 7. Save JSON ──────────────────────────────────────────────────────────────
def make_serial(obj):
    if isinstance(obj, (np.floating, float)): return float(obj)
    if isinstance(obj, (np.integer, int)):    return int(obj)
    if isinstance(obj, dict):  return {k: make_serial(v) for k, v in obj.items()}
    if isinstance(obj, list):  return [make_serial(x) for x in obj]
    return obj

results = {
    "oracle_val_wrmsse":  wrmsse,
    "baseline_wrmsse":    0.5422,
    "delta":              wrmsse - 0.5422,
    "best_iteration":     model.best_iteration,
    "train_time_s":       train_time,
    "feature_importance": importance[["feature","rank","gain","split"]].to_dict("records"),
    "yearly_lag_ranks":   {row["feature"]: int(row["rank"]) for _, row in yearly_lags.iterrows()},
    "verdict":            verdict,
}
json_path = os.path.join(REPORTS, "day15_sanity_lags_scores.json")
with open(json_path, "w") as f:
    json.dump(make_serial(results), f, indent=2)
print(f"\n    Saved {json_path}")

# ── summary ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STAGE A SUMMARY")
print("=" * 65)
print(f"\n  Training time:       {train_time:.0f}s ({train_time/60:.1f} min)")
print(f"  Oracle val WRMSSE:   {wrmsse:.4f}  (baseline 0.5422, delta {wrmsse-0.5422:+.4f})")
print(f"  Best iter:           {model.best_iteration}")
print(f"\n  Yearly lag ranks (out of {len(ALL_FEATURES)}):")
for _, row in yearly_lags.iterrows():
    print(f"    {row['feature']}: rank {int(row['rank'])}")
print(f"\n  Stage B verdict: {verdict}")
print(f"  {reason}")
print("\nDone!")
