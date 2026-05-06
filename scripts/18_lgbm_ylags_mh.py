"""
WS2.5 Final Variant — Annual Lags + tvp=1.3 Multi-Horizon

28 direct-horizon LightGBM models, Day 9 params, tvp=1.3.
Feature set extends the tvp=1.3 baseline (scripts/12) with lag_91, lag_182, lag_364.
Same-origin val scoring from d_1913. No recursive compounding.

After training:
  - Submit standalone to Kaggle
  - Optuna 50-trial ensemble: {ylags_mh, tvp_1p3, store_dept}
  - Submit ensemble if val improvement > 0.005 over best component
"""
from __future__ import annotations
import sys, os, time, json, pickle, gc, hashlib, warnings, subprocess
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

# ── Design fingerprint (change any value → new hash → retrain forced) ─────────
_DESIGN = {
    "objective":   "tweedie",
    "tvp":         1.3,
    "feat_start":  1000,
    "last_train":  1913,
    "val_start":   1886,
    "feature_set": "v1_41num_4cat_ylags",
}
DESIGN_HASH = hashlib.md5(json.dumps(_DESIGN, sort_keys=True).encode()).hexdigest()[:8]

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_RAW  = os.path.join(PROJ_ROOT, "data", "raw", "m5-forecasting-accuracy")
FEAT_DIR  = os.path.join(PROJ_ROOT, "data", "processed", "features")
MODEL_DIR = os.path.join(PROJ_ROOT, "data", "models", "tvp_1p3_ylags")
PREDS_DIR = os.path.join(PROJ_ROOT, "data", "predictions")
REPORTS   = os.path.join(PROJ_ROOT, "reports")
SUBS      = os.path.join(PROJ_ROOT, "submissions")
for d in [MODEL_DIR, PREDS_DIR, SUBS]:
    os.makedirs(d, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
LAST_TRAIN = 1913
VAL_START  = 1886
FEAT_START = 1000
HORIZON    = 28

CAT_FEATURES = ["cat_id", "dept_id", "store_id", "state_id"]
NUM_FEATURES = [
    "weekday", "month", "quarter", "year", "day_of_month", "week_of_year",
    "is_weekend", "is_holiday",
    "is_snap_ca", "is_snap_tx", "is_snap_wi",
    "days_since_event", "days_until_next_event",
    "sell_price", "price_change_pct", "price_relative_mean",
    "price_volatility", "has_price_change",
    "lag_7", "lag_14", "lag_28", "lag_56",
    "lag_91", "lag_182", "lag_364",
    "roll_mean_7",  "roll_std_7",  "roll_min_7",  "roll_max_7",
    "roll_mean_28", "roll_std_28", "roll_min_28", "roll_max_28",
    "roll_mean_56", "roll_std_56", "roll_min_56", "roll_max_56",
    "roll_mean_180","roll_std_180","roll_min_180","roll_max_180",
]
ALL_FEATURES = NUM_FEATURES + CAT_FEATURES  # 45 total

# Day 9 Optuna best params, tvp=1.3
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
    "tweedie_variance_power": 1.3,
}

print("=" * 65)
print("WS2.5 Final — Annual Lags + tvp=1.3 Multi-Horizon")
print(f"Design hash: {DESIGN_HASH}   Features: {len(ALL_FEATURES)}")
print(f"New features: lag_91, lag_182, lag_364")
print(f"Model dir: {MODEL_DIR}")
print("=" * 65)

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
print(f"    Rows: {len(df):,}  Cols: {len(df.columns)}  ({time.time()-t0:.1f}s)")

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
print(f"\n[3] Training 28 models (annual lags, tvp=1.3)...")
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

# ── 4. Val WRMSSE (same-origin from d_1913) ───────────────────────────────────
print(f"\n[4] Val WRMSSE (same-origin from d_{LAST_TRAIN})...")
val_preds = np.zeros((n_series, HORIZON), dtype=np.float32)
for h in range(1, HORIZON + 1):
    val_preds[:, h-1] = np.clip(models[h].predict(df_val_origin[ALL_FEATURES]), 0.0, None)

val_wrmsse = score_preds(val_preds, series_ids)
print(f"    ylags_mh val WRMSSE:  {val_wrmsse:.4f}")
print(f"    tvp=1.3 baseline:     0.6860")
print(f"    Delta vs tvp=1.3:     {val_wrmsse - 0.6860:+.4f}")

# ── 5. Eval predictions from d_1941 ──────────────────────────────────────────
print(f"\n[5] Eval predictions (from d_{LAST_TRAIN + HORIZON})...")
eval_preds = np.zeros((n_series, HORIZON), dtype=np.float32)
for h in range(1, HORIZON + 1):
    eval_preds[:, h-1] = np.clip(models[h].predict(df_eval_origin[ALL_FEATURES]), 0.0, None)

# ── 6. Save prediction parquets ──────────────────────────────────────────────
print(f"\n[6] Saving prediction parquets...")
val_df = pd.DataFrame(val_preds, columns=f_cols)
val_df.insert(0, "id", series_ids)
val_df.to_parquet(os.path.join(PREDS_DIR, "lgbm_ylags_mh_val.parquet"), index=False)

eval_df = pd.DataFrame(eval_preds, columns=f_cols)
eval_df.insert(0, "id", series_ids)
eval_df.to_parquet(os.path.join(PREDS_DIR, "lgbm_ylags_mh_eval.parquet"), index=False)
print("    Saved lgbm_ylags_mh_val.parquet and lgbm_ylags_mh_eval.parquet")

# ── 7. Build submission CSV ───────────────────────────────────────────────────
print(f"\n[7] Building submission (lgbm_ylags_mh.csv)...")
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


build_submission(val_preds, eval_preds, series_ids, "lgbm_ylags_mh.csv")

# ── 8. Submit standalone to Kaggle ────────────────────────────────────────────
print(f"\n[8] Submitting standalone to Kaggle...")
result = subprocess.run([
    "kaggle", "competitions", "submit",
    "-c", "m5-forecasting-accuracy",
    "-f", os.path.join(SUBS, "lgbm_ylags_mh.csv"),
    "-m", f"WS2.5 Final: annual lags (lag_91/182/364) + tvp=1.3 MH  hash={DESIGN_HASH}",
], capture_output=True, text=True)
print(result.stdout.strip())
if result.returncode != 0:
    print(f"WARN: Kaggle submit stderr: {result.stderr.strip()}")

# ── 9. Optuna ensemble: {ylags_mh, tvp_1p3, store_dept} ──────────────────────
print(f"\n[9] Optuna 50-trial ensemble {{ylags_mh, tvp_1p3, store_dept}}...")
val_tvp13 = pd.read_parquet(os.path.join(PREDS_DIR, "lgbm_tvp_1p3_val.parquet")).sort_values("id").reset_index(drop=True)
val_sd    = pd.read_parquet(os.path.join(PREDS_DIR, "lgbm_store_dept_val.parquet")).sort_values("id").reset_index(drop=True)
val_yl    = pd.read_parquet(os.path.join(PREDS_DIR, "lgbm_ylags_mh_val.parquet")).sort_values("id").reset_index(drop=True)

assert (val_tvp13["id"] == val_yl["id"]).all(), "Series mismatch: tvp13 vs ylags"
assert (val_sd["id"]    == val_yl["id"]).all(), "Series mismatch: sd vs ylags"

ens_sids   = val_yl["id"].values
mat_tvp13  = val_tvp13[f_cols].values.astype(np.float32)
mat_sd     = val_sd[f_cols].values.astype(np.float32)
mat_yl     = val_yl[f_cols].values.astype(np.float32)

sub_ens    = sales_eval[sales_eval["id"].isin(ens_sids)].set_index("id").reindex(ens_sids).reset_index()
acts_ens   = sub_ens[actual_val_cols].values.astype(np.float32)


def ens_wrmsse(mat):
    s, _ = compute_wrmsse(
        preds=mat, actuals=acts_ens,
        sales_df=sub_ens, prices_df=prices_df,
        calendar_df=calendar_df, last_train_day=LAST_TRAIN,
    )
    return float(s)


val_yl_score    = val_wrmsse
val_tvp13_score = ens_wrmsse(mat_tvp13)
val_sd_score    = ens_wrmsse(mat_sd)
best_component  = min(val_yl_score, val_tvp13_score, val_sd_score)
print(f"    Component val WRMSSEs:")
print(f"      ylags_mh:   {val_yl_score:.4f}")
print(f"      tvp_1p3:    {val_tvp13_score:.4f}")
print(f"      store_dept: {val_sd_score:.4f}")
print(f"    Best component: {best_component:.4f}")


def objective(trial):
    w_yl  = trial.suggest_float("w_yl",  0.0, 1.0)
    w_t13 = trial.suggest_float("w_t13", 0.0, 1.0 - w_yl)
    w_sd  = 1.0 - w_yl - w_t13
    return ens_wrmsse(w_yl * mat_yl + w_t13 * mat_tvp13 + w_sd * mat_sd)


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50, show_progress_bar=False)

best_val    = study.best_value
best_w_yl   = study.best_params["w_yl"]
best_w_t13  = study.best_params["w_t13"]
best_w_sd   = 1.0 - best_w_yl - best_w_t13
improvement = best_component - best_val

print(f"\n    Best ensemble val WRMSSE: {best_val:.4f}")
print(f"    Weights: ylags={best_w_yl:.3f}  tvp13={best_w_t13:.3f}  sd={best_w_sd:.3f}")
print(f"    Improvement vs best component: {improvement:+.4f}")
SUBMIT_ENS = improvement > 0.005
print(f"    Submit ensemble: {'YES' if SUBMIT_ENS else 'NO'} (threshold: >0.005)")

# ── 10. Build ensemble CSV (and optionally submit) ────────────────────────────
eval_tvp13 = pd.read_parquet(os.path.join(PREDS_DIR, "lgbm_tvp_1p3_eval.parquet")).sort_values("id").reset_index(drop=True)
eval_sd    = pd.read_parquet(os.path.join(PREDS_DIR, "lgbm_store_dept_eval.parquet")).sort_values("id").reset_index(drop=True)
eval_yl    = pd.read_parquet(os.path.join(PREDS_DIR, "lgbm_ylags_mh_eval.parquet")).sort_values("id").reset_index(drop=True)

ens_val_mat  = best_w_yl * mat_yl + best_w_t13 * mat_tvp13 + best_w_sd * mat_sd
ens_eval_mat = (
    best_w_yl  * eval_yl[f_cols].values.astype(np.float32)  +
    best_w_t13 * eval_tvp13[f_cols].values.astype(np.float32) +
    best_w_sd  * eval_sd[f_cols].values.astype(np.float32)
)

print(f"\n[10] Building ensemble submission (lgbm_ylags_ensemble.csv)...")
build_submission(ens_val_mat, ens_eval_mat, ens_sids, "lgbm_ylags_ensemble.csv")

if SUBMIT_ENS:
    print(f"\n[10b] Submitting ensemble to Kaggle...")
    result2 = subprocess.run([
        "kaggle", "competitions", "submit",
        "-c", "m5-forecasting-accuracy",
        "-f", os.path.join(SUBS, "lgbm_ylags_ensemble.csv"),
        "-m", (f"WS2.5 Final ensemble: ylags×{best_w_yl:.3f} + "
               f"tvp13×{best_w_t13:.3f} + sd×{best_w_sd:.3f}  hash={DESIGN_HASH}"),
    ], capture_output=True, text=True)
    print(result2.stdout.strip())
    if result2.returncode != 0:
        print(f"WARN: ensemble submit stderr: {result2.stderr.strip()}")
else:
    print("    Ensemble CSV saved but not submitted (improvement <= 0.005).")

# ── 11. Save results JSON ─────────────────────────────────────────────────────
def make_serial(obj):
    if isinstance(obj, (np.floating, float)): return float(obj)
    if isinstance(obj, (np.integer, int)):    return int(obj)
    if isinstance(obj, dict):  return {k: make_serial(v) for k, v in obj.items()}
    if isinstance(obj, list):  return [make_serial(x) for x in obj]
    return obj


results = {
    "variant":         "ylags_mh",
    "design_hash":     DESIGN_HASH,
    "feature_count":   len(ALL_FEATURES),
    "new_features":    ["lag_91", "lag_182", "lag_364"],
    "total_train_min": total_train_min,
    "val_wrmsse":      val_wrmsse,
    "ref_tvp13_val":   0.6860,
    "ref_sd_val":      0.6294,
    "delta_vs_tvp13":  val_wrmsse - 0.6860,
    "ensemble": {
        "best_val":    best_val,
        "improvement": improvement,
        "submitted":   SUBMIT_ENS,
        "weights": {
            "ylags_mh":   best_w_yl,
            "tvp_1p3":    best_w_t13,
            "store_dept": best_w_sd,
        },
    },
    "per_horizon": {str(h): h_scores.get(h, {}) for h in range(1, HORIZON + 1)},
}

scores_path = os.path.join(REPORTS, "day18_ylags_scores.json")
with open(scores_path, "w") as f:
    json.dump(make_serial(results), f, indent=2)
print(f"\n    Saved {scores_path}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("WS2.5 FINAL — ANNUAL LAGS SUMMARY")
print("=" * 65)
print(f"\nDesign hash:    {DESIGN_HASH}")
print(f"Training time:  {total_train_min:.1f} min  (28 models)")
print(f"Val WRMSSE:     {val_wrmsse:.4f}  (tvp=1.3 ref: 0.6860, delta: {val_wrmsse-0.6860:+.4f})")
print(f"\nEnsemble best:  {best_val:.4f}  (improvement vs best: {improvement:+.4f})")
print(f"Ens submitted:  {'YES' if SUBMIT_ENS else 'NO'}")
print(f"\nOutputs:")
print(f"  data/models/tvp_1p3_ylags/  (28 pkl files)")
print(f"  data/predictions/lgbm_ylags_mh_{{val,eval}}.parquet")
print(f"  submissions/lgbm_ylags_mh.csv")
print(f"  submissions/lgbm_ylags_ensemble.csv")
print(f"  reports/day18_ylags_scores.json")
print("\nDone!")
