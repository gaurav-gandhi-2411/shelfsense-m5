"""
Day 6: LightGBM global model training.

Stages:
  1. Load features (d_num >= 1000)
  2. RMSE baseline model
  3. Tweedie variant
  4. Optuna search on d_num >= 1600 subset
  5. Best model retrain on full d_num >= 1000
  6. Feature importance
  7. Per-category WRMSSE breakdown
  8. Kaggle submission build
"""
from __future__ import annotations

import sys, os, time, json, random, warnings, pickle
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

print("=" * 60)
print("DAY 6 -- LightGBM Global Model")
print("=" * 60)

# ── paths ──────────────────────────────────────────────────────────────────────
DATA_RAW  = os.path.join(PROJ_ROOT, "data", "raw", "m5-forecasting-accuracy")
FEAT_DIR  = os.path.join(PROJ_ROOT, "data", "processed", "features")
MODELS    = os.path.join(PROJ_ROOT, "data", "models")
REPORTS   = os.path.join(PROJ_ROOT, "reports")
CHARTS    = os.path.join(REPORTS, "charts")
SUBS      = os.path.join(PROJ_ROOT, "submissions")
os.makedirs(MODELS, exist_ok=True)
os.makedirs(CHARTS, exist_ok=True)

LAST_TRAIN    = 1913
VAL_START     = 1886   # last 28 training days = validation window
FEAT_START    = 1000   # drop very early rows (unstable lag features)
OPTUNA_START  = 1600   # smaller subset for Optuna speed
HORIZON       = 28

# ── feature columns ────────────────────────────────────────────────────────────
CAT_FEATURES = ["cat_id", "dept_id", "store_id", "state_id"]
NUM_FEATURES = [
    # calendar
    "weekday", "month", "quarter", "year", "day_of_month", "week_of_year",
    "is_weekend", "is_holiday",
    "is_snap_ca", "is_snap_tx", "is_snap_wi",
    "days_since_event", "days_until_next_event",
    # price
    "sell_price", "price_change_pct", "price_relative_mean",
    "price_volatility", "has_price_change",
    # lags
    "lag_7", "lag_14", "lag_28", "lag_56",
    # rolling
    "roll_mean_7",  "roll_std_7",  "roll_min_7",  "roll_max_7",
    "roll_mean_28", "roll_std_28", "roll_min_28", "roll_max_28",
    "roll_mean_56", "roll_std_56", "roll_min_56", "roll_max_56",
    "roll_mean_180","roll_std_180","roll_min_180","roll_max_180",
]
ALL_FEATURES = NUM_FEATURES + CAT_FEATURES
META = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]

# ── 1. Load features ───────────────────────────────────────────────────────────
print(f"\n[1] Loading features (d_num >= {FEAT_START})...")
t0 = time.time()
df = pd.read_parquet(
    FEAT_DIR,
    filters=[("d_num", ">=", FEAT_START)],
    columns=["id", "item_id", "cat_id", "dept_id", "store_id", "state_id",
             "d_num", "sales"] + NUM_FEATURES,
)
print(f"    Loaded: {df.shape}  ({time.time()-t0:.1f}s)")
print(f"    Memory: {df.memory_usage(deep=True).sum()/1e9:.2f} GB")

# Re-attach categorical dtypes (parquet may not preserve after filter)
from features.hierarchy import CAT_DTYPES
for col, dtype in CAT_DTYPES.items():
    if col in df.columns:
        df[col] = df[col].astype(dtype)

# Drop rows where any lag is NaN (first few days of a series)
lag_cols = ["lag_7", "lag_14", "lag_28", "lag_56"]
df = df.dropna(subset=lag_cols).reset_index(drop=True)
print(f"    After dropping lag-NaN rows: {len(df):,}")

# ── 2. Train / val split ────────────────────────────────────────────────────────
print(f"\n[2] Train/val split (train d_num <= {VAL_START-1}, val {VAL_START}-{LAST_TRAIN})...")
train_mask = df["d_num"] <= VAL_START - 1
val_mask   = (df["d_num"] >= VAL_START) & (df["d_num"] <= LAST_TRAIN)

X_train = df.loc[train_mask, ALL_FEATURES]
y_train = df.loc[train_mask, "sales"].astype(np.float32)
X_val   = df.loc[val_mask,   ALL_FEATURES]
y_val   = df.loc[val_mask,   "sales"].astype(np.float32)

print(f"    Train: {len(X_train):,} rows  Val: {len(X_val):,} rows")

# Build LightGBM datasets
print("    Building LightGBM datasets...")
t0 = time.time()
lgb_train = lgb.Dataset(
    X_train, label=y_train,
    categorical_feature=CAT_FEATURES,
    free_raw_data=False,
)
lgb_val = lgb.Dataset(
    X_val, label=y_val,
    categorical_feature=CAT_FEATURES,
    reference=lgb_train,
    free_raw_data=False,
)
print(f"    Dataset built in {time.time()-t0:.1f}s")

# ── helper: generate forecasts from trained model ──────────────────────────────
def generate_val_preds(model, df_full):
    """Predict over the validation window (d_num 1914-1941) using pre-computed features."""
    fcast_mask = (df_full["d_num"] >= LAST_TRAIN + 1) & \
                 (df_full["d_num"] <= LAST_TRAIN + HORIZON)
    df_fcast = df_full[fcast_mask].copy()
    if len(df_fcast) == 0:
        # Forecast features not in current slice — reload them
        return None, None
    df_fcast = df_fcast.sort_values(["id", "d_num"]).reset_index(drop=True)
    preds_flat = model.predict(df_fcast[ALL_FEATURES])
    preds_flat = np.clip(preds_flat, 0, None).astype(np.float32)
    series_ids = df_fcast["id"].unique()
    n_series   = len(series_ids)
    preds_mat  = preds_flat.reshape(n_series, HORIZON)
    return preds_mat, df_fcast, series_ids

def score_preds(preds_mat, series_ids, sales_eval, prices_df, calendar_df):
    from evaluation.wrmsse import compute_wrmsse
    sales_sub = (
        sales_eval[sales_eval["id"].isin(series_ids)]
        .set_index("id").reindex(series_ids).reset_index()
    )
    actual_cols = [f"d_{d}" for d in range(LAST_TRAIN+1, LAST_TRAIN+HORIZON+1)]
    actuals = sales_sub[actual_cols].values.astype(np.float32)
    score, levels = compute_wrmsse(
        preds=preds_mat, actuals=actuals,
        sales_df=sales_sub, prices_df=prices_df,
        calendar_df=calendar_df, last_train_day=LAST_TRAIN,
    )
    return score, levels

# Load raw data for WRMSSE scoring
print("\n[3] Loading raw sales/prices/calendar for WRMSSE...")
t0 = time.time()
sales_eval  = pd.read_csv(os.path.join(DATA_RAW, "sales_train_evaluation.csv"))
prices_df   = pd.read_csv(os.path.join(DATA_RAW, "sell_prices.csv"))
calendar_df = pd.read_csv(os.path.join(DATA_RAW, "calendar.csv"))
print(f"    Loaded in {time.time()-t0:.1f}s")

# Also load the forecast features (d_num 1914-1941) — separate read since filter above excluded them
print("    Loading forecast features (d_num 1914-1941)...")
df_fcast = pd.read_parquet(
    FEAT_DIR,
    filters=[("d_num", ">=", LAST_TRAIN + 1), ("d_num", "<=", LAST_TRAIN + HORIZON)],
    columns=["id", "item_id", "cat_id", "dept_id", "store_id", "state_id",
             "d_num", "sales"] + NUM_FEATURES,
)
for col, dtype in CAT_DTYPES.items():
    if col in df_fcast.columns:
        df_fcast[col] = df_fcast[col].astype(dtype)
df_fcast = df_fcast.sort_values(["id", "d_num"]).reset_index(drop=True)
print(f"    Forecast features: {df_fcast.shape}")

def predict_validation(model):
    """Predict d_1914-d_1941 for all 30,490 series."""
    preds_flat = model.predict(df_fcast[ALL_FEATURES])
    preds_flat = np.clip(preds_flat, 0, None).astype(np.float32)
    # Reshape: df_fcast is sorted by (id, d_num)
    n_series = len(df_fcast["id"].unique())
    return preds_flat.reshape(n_series, HORIZON)

def get_series_ids():
    return df_fcast.sort_values(["id", "d_num"]).groupby("id").first().reset_index()["id"].values

# ── base params ────────────────────────────────────────────────────────────────
BASE_PARAMS = {
    "learning_rate":    0.075,
    "num_leaves":       128,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.75,
    "bagging_freq":     1,
    "lambda_l2":        0.1,
    "verbose":         -1,
    "num_threads":      0,     # use all available cores
    "seed":             42,
}

CB_EARLY = lgb.early_stopping(50, verbose=False)
CB_LOG   = lgb.log_evaluation(100)

all_scores = {}

# ── 4. RMSE model ──────────────────────────────────────────────────────────────
print("\n[4] Training RMSE model...")
t0 = time.time()
params_rmse = {**BASE_PARAMS, "objective": "regression", "metric": "rmse"}
model_rmse = lgb.train(
    params_rmse, lgb_train,
    num_boost_round=2000,
    valid_sets=[lgb_val],
    callbacks=[CB_EARLY, CB_LOG],
)
wall_rmse = time.time() - t0
print(f"    Best iteration: {model_rmse.best_iteration}  Time: {wall_rmse:.0f}s")

preds_rmse = predict_validation(model_rmse)
series_ids = get_series_ids()
wrmsse_rmse, levels_rmse = score_preds(preds_rmse, series_ids, sales_eval, prices_df, calendar_df)
print(f"    WRMSSE (full 30490 series): {wrmsse_rmse:.4f}")
all_scores["rmse"] = {"wrmsse": float(wrmsse_rmse), "best_iter": model_rmse.best_iteration}

with open(os.path.join(MODELS, "lgbm_rmse.pkl"), "wb") as f:
    pickle.dump(model_rmse, f)

# ── 5. Tweedie model ───────────────────────────────────────────────────────────
print("\n[5] Training Tweedie model (variance_power=1.1)...")
t0 = time.time()
params_tw = {
    **BASE_PARAMS,
    "objective": "tweedie",
    "metric": "tweedie",
    "tweedie_variance_power": 1.1,
}
model_tw = lgb.train(
    params_tw, lgb_train,
    num_boost_round=2000,
    valid_sets=[lgb_val],
    callbacks=[CB_EARLY, CB_LOG],
)
wall_tw = time.time() - t0
print(f"    Best iteration: {model_tw.best_iteration}  Time: {wall_tw:.0f}s")

preds_tw = predict_validation(model_tw)
wrmsse_tw, levels_tw = score_preds(preds_tw, series_ids, sales_eval, prices_df, calendar_df)
print(f"    WRMSSE (full 30490 series): {wrmsse_tw:.4f}")
all_scores["tweedie"] = {"wrmsse": float(wrmsse_tw), "best_iter": model_tw.best_iteration}

with open(os.path.join(MODELS, "lgbm_tweedie.pkl"), "wb") as f:
    pickle.dump(model_tw, f)

print(f"\n    RMSE vs Tweedie: {wrmsse_rmse:.4f} vs {wrmsse_tw:.4f}")
best_loss = "tweedie" if wrmsse_tw < wrmsse_rmse else "rmse"
print(f"    Best loss for Optuna: {best_loss}")

# ── 6. Optuna hyperparameter search ─────────────────────────────────────────────
print(f"\n[6] Optuna search ({20} trials, d_num >= {OPTUNA_START})...")

# Subset dataset for fast trials
opt_train_mask = (df["d_num"] >= OPTUNA_START) & (df["d_num"] <= VAL_START - 1)
opt_val_mask   = (df["d_num"] >= VAL_START) & (df["d_num"] <= LAST_TRAIN)

X_opt_train = df.loc[opt_train_mask, ALL_FEATURES]
y_opt_train = df.loc[opt_train_mask, "sales"].astype(np.float32)
X_opt_val   = df.loc[opt_val_mask,   ALL_FEATURES]
y_opt_val   = df.loc[opt_val_mask,   "sales"].astype(np.float32)
print(f"    Optuna train: {len(X_opt_train):,}  val: {len(X_opt_val):,} rows")

lgb_opt_train = lgb.Dataset(X_opt_train, label=y_opt_train, categorical_feature=CAT_FEATURES, free_raw_data=False)
lgb_opt_val   = lgb.Dataset(X_opt_val,   label=y_opt_val,   categorical_feature=CAT_FEATURES, reference=lgb_opt_train, free_raw_data=False)

BEST_OBJ   = "tweedie" if best_loss == "tweedie" else "regression"
BEST_METRIC= "tweedie" if best_loss == "tweedie" else "rmse"
BEST_TVP   = 1.1 if best_loss == "tweedie" else None

def optuna_objective(trial):
    p = {
        "objective":        BEST_OBJ,
        "metric":           BEST_METRIC,
        "verbose":         -1,
        "seed":             42,
        "num_threads":      0,
        "bagging_freq":     1,
        "learning_rate":    trial.suggest_categorical("lr", [0.025, 0.05, 0.075, 0.1]),
        "num_leaves":       trial.suggest_categorical("num_leaves", [32, 64, 128, 256]),
        "min_data_in_leaf": trial.suggest_categorical("min_data", [20, 50, 100, 200]),
        "feature_fraction": trial.suggest_categorical("feat_frac", [0.5, 0.7, 0.8, 0.9]),
        "bagging_fraction": trial.suggest_categorical("bag_frac",  [0.5, 0.7, 0.8, 0.9]),
        "lambda_l2":        trial.suggest_categorical("l2", [0.0, 0.1, 0.5, 1.0]),
    }
    if BEST_TVP is not None:
        p["tweedie_variance_power"] = trial.suggest_float("tvp", 1.0, 1.5)

    m = lgb.train(
        p, lgb_opt_train,
        num_boost_round=500,
        valid_sets=[lgb_opt_val],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
    )
    # Return best val metric from LightGBM (lower is better)
    return m.best_score["valid_0"][BEST_METRIC]

study = optuna.create_study(direction="minimize")
t0 = time.time()
study.optimize(optuna_objective, n_trials=20, show_progress_bar=False)
optuna_time = time.time() - t0
print(f"    Optuna done in {optuna_time:.0f}s  Best value: {study.best_value:.4f}")
print(f"    Best params: {study.best_params}")
all_scores["optuna_best_params"] = study.best_params
all_scores["optuna_best_val_metric"] = float(study.best_value)

# ── 7. Retrain best model on full d_num >= FEAT_START ─────────────────────────
print(f"\n[7] Retraining best model on full d_num >= {FEAT_START}...")
best_p = {
    **BASE_PARAMS,
    "objective":        BEST_OBJ,
    "metric":           BEST_METRIC,
    "learning_rate":    study.best_params.get("lr",         BASE_PARAMS["learning_rate"]),
    "num_leaves":       study.best_params.get("num_leaves", BASE_PARAMS["num_leaves"]),
    "min_data_in_leaf": study.best_params.get("min_data",   BASE_PARAMS["min_data_in_leaf"]),
    "feature_fraction": study.best_params.get("feat_frac",  BASE_PARAMS["feature_fraction"]),
    "bagging_fraction": study.best_params.get("bag_frac",   BASE_PARAMS["bagging_fraction"]),
    "lambda_l2":        study.best_params.get("l2",         BASE_PARAMS["lambda_l2"]),
}
if BEST_TVP is not None:
    best_p["tweedie_variance_power"] = study.best_params.get("tvp", BEST_TVP)

# Use best_iteration from initial model × 1.1 as fixed rounds (no val set, train on all)
# Actually retrain WITH val set for early stopping consistency
t0 = time.time()
model_best = lgb.train(
    best_p, lgb_train,
    num_boost_round=3000,
    valid_sets=[lgb_val],
    callbacks=[lgb.early_stopping(75, verbose=False), CB_LOG],
)
wall_best = time.time() - t0
print(f"    Best iteration: {model_best.best_iteration}  Time: {wall_best:.0f}s")

preds_best = predict_validation(model_best)
wrmsse_best, levels_best = score_preds(preds_best, series_ids, sales_eval, prices_df, calendar_df)
print(f"    WRMSSE (full 30490 series): {wrmsse_best:.4f}")
all_scores["best_tuned"] = {
    "wrmsse": float(wrmsse_best),
    "best_iter": model_best.best_iteration,
    "params": best_p,
}

with open(os.path.join(MODELS, "lgbm_best.pkl"), "wb") as f:
    pickle.dump(model_best, f)

# ── 8. Feature importance ──────────────────────────────────────────────────────
print("\n[8] Feature importance...")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

importance = pd.DataFrame({
    "feature": model_best.feature_name(),
    "gain":    model_best.feature_importance(importance_type="gain"),
    "split":   model_best.feature_importance(importance_type="split"),
}).sort_values("gain", ascending=False).reset_index(drop=True)

print(f"    Top 20 features (by gain):")
print(importance.head(20).to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 8))
top20 = importance.head(20)
bars = ax.barh(top20["feature"][::-1], top20["gain"][::-1], color="steelblue")
ax.set_xlabel("Feature Importance (Gain)")
ax.set_title("LightGBM - Top 20 Features by Gain")
plt.tight_layout()
plt.savefig(os.path.join(CHARTS, "lgbm_feature_importance.png"), dpi=120, bbox_inches="tight")
plt.close()
print(f"    Saved: reports/charts/lgbm_feature_importance.png")

# ── 9. Per-category WRMSSE ────────────────────────────────────────────────────
print("\n[9] Per-category WRMSSE breakdown...")
from evaluation.wrmsse import compute_wrmsse

# Build prediction matrix aligned to sales_eval order
pred_series_order = df_fcast.sort_values(["id", "d_num"]).groupby("id", sort=True).first().reset_index()["id"].values
preds_all_flat = model_best.predict(df_fcast.sort_values(["id", "d_num"])[ALL_FEATURES])
preds_all_flat = np.clip(preds_all_flat, 0, None).astype(np.float32)
preds_all_mat  = preds_all_flat.reshape(len(pred_series_order), HORIZON)

actual_cols = [f"d_{d}" for d in range(LAST_TRAIN + 1, LAST_TRAIN + HORIZON + 1)]

cat_results = {}
for cat in ["FOODS", "HOUSEHOLD", "HOBBIES"]:
    cat_ids = [sid for sid in pred_series_order if cat in sid]
    idx = [i for i, sid in enumerate(pred_series_order) if cat in sid]
    sub_sales = sales_eval[sales_eval["id"].isin(cat_ids)].set_index("id").reindex(cat_ids).reset_index()
    sub_preds  = preds_all_mat[idx]
    sub_acts   = sub_sales[actual_cols].values.astype(np.float32)
    s, _ = compute_wrmsse(preds=sub_preds, actuals=sub_acts,
                          sales_df=sub_sales, prices_df=prices_df,
                          calendar_df=calendar_df, last_train_day=LAST_TRAIN)
    cat_results[cat] = float(s)
    print(f"    {cat}: WRMSSE = {s:.4f}")

all_scores["best_by_category"] = cat_results

# ── 10. Build Kaggle submission ────────────────────────────────────────────────
print("\n[10] Building Kaggle submissions...")

# Full-catalogue predictions for validation rows (d_1914-d_1941 = _validation suffix)
# Use SN28 for the _evaluation rows (d_1942-d_1969)
sn28_base = pd.read_csv(os.path.join(SUBS, "seasonal_naive_28_submission.csv"))

def build_full_submission(preds_mat, series_ids, fname):
    """Build 60,980-row Kaggle submission: LightGBM for val rows, SN28 for eval rows."""
    f_cols = [f"F{i}" for i in range(1, HORIZON + 1)]
    val_df = pd.DataFrame(preds_mat, columns=f_cols)
    # Map to _validation suffix IDs
    val_ids = [sid.replace("_evaluation", "_validation") for sid in series_ids]
    val_df.insert(0, "id", val_ids)

    full = sn28_base.copy().set_index("id")
    val_indexed = val_df.set_index("id")
    overlap = val_indexed.index.intersection(full.index)
    full.loc[overlap] = val_indexed.loc[overlap]
    full = full.reset_index()
    path = os.path.join(SUBS, fname)
    full.to_csv(path, index=False)
    print(f"    Saved: {path}  ({len(overlap)} val rows overridden)")
    return path

for label, preds_m, model_k in [
    ("rmse",    preds_rmse, "rmse"),
    ("tweedie", preds_tw,   "tweedie"),
    ("best",    preds_best, "best"),
]:
    build_full_submission(preds_m, series_ids, f"lgbm_{label}_submission.csv")

# ── 11. Save results JSON ──────────────────────────────────────────────────────
def make_serial(obj):
    if isinstance(obj, (np.floating, float)): return float(obj)
    if isinstance(obj, (np.integer, int)):    return int(obj)
    if isinstance(obj, dict):  return {k: make_serial(v) for k, v in obj.items()}
    if isinstance(obj, list):  return [make_serial(x) for x in obj]
    return obj

results_path = os.path.join(REPORTS, "day6_lgbm_scores.json")
with open(results_path, "w") as f:
    json.dump(make_serial(all_scores), f, indent=2)
print(f"\n    Saved scores: {results_path}")
importance.to_csv(os.path.join(REPORTS, "day6_feature_importance.csv"), index=False)

# ── summary ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("DAY 6 SUMMARY")
print("=" * 60)
print(f"\nModel             WRMSSE (full 30490)")
print(f"  RMSE baseline:  {wrmsse_rmse:.4f}")
print(f"  Tweedie 1.1:    {wrmsse_tw:.4f}")
print(f"  Best tuned:     {wrmsse_best:.4f}")
print(f"\nBest model: loss={best_loss}  Optuna best trial: {study.best_value:.4f}")
print(f"\nPer-category (best model):")
for cat, s in cat_results.items():
    print(f"  {cat}: {s:.4f}")
print(f"\nTop 5 features by gain:")
for _, row in importance.head(5).iterrows():
    print(f"  {row['feature']}: {row['gain']:.0f}")
print("\nDone!")
