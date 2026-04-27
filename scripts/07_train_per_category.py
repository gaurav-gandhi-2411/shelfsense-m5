"""
Day 7: Per-category LightGBM models + recursive evaluation period forecast.

Stages:
  1. Load features parquet + raw CSVs + Day 6 global model
  2. Sanity check: recursive forecast on validation period (global model)
     — confirm recursive WRMSSE ≈ single-step 0.5422 (within ~5%)
  3. Per-category Optuna (FOODS / HOUSEHOLD / HOBBIES) + retrain
  4. Per-category validation scores
  5. Recursive evaluation period forecast (d_1942–d_1969):
     a. Global model
     b. Per-category models (each category uses its own model)
  6. Blend: 0.6 × per_category + 0.4 × global (val + eval)
  7. Build Kaggle submissions:
     a. lgbm_global_recursive.csv
     b. lgbm_per_cat.csv
     c. lgbm_blend.csv
  8. Save results JSON + write reports/06_per_category_models.md
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
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from features.hierarchy import CAT_DTYPES
from features.calendar import build_calendar_lookup
from features.price import build_price_lookup
from models.recursive_forecast import (
    build_sales_buffer, build_eval_price_features, predict_recursive,
    ALL_FEATURES, NUM_FEATURES, CAT_COLS,
)
from evaluation.wrmsse import compute_wrmsse

print("=" * 60)
print("DAY 7 -- Per-Category LightGBM + Recursive Eval Forecast")
print("=" * 60)

# ── paths ──────────────────────────────────────────────────────────────────────
DATA_RAW  = os.path.join(PROJ_ROOT, "data", "raw", "m5-forecasting-accuracy")
FEAT_DIR  = os.path.join(PROJ_ROOT, "data", "processed", "features")
MODELS    = os.path.join(PROJ_ROOT, "data", "models")
REPORTS   = os.path.join(PROJ_ROOT, "reports")
SUBS      = os.path.join(PROJ_ROOT, "submissions")
os.makedirs(MODELS, exist_ok=True)

LAST_TRAIN   = 1913
VAL_START    = 1886
FEAT_START   = 1000
OPTUNA_START = 1700
HORIZON      = 28
EVAL_START   = LAST_TRAIN + HORIZON + 1   # 1942
EVAL_END     = EVAL_START + HORIZON - 1   # 1969
CATEGORIES   = ["FOODS", "HOUSEHOLD", "HOBBIES"]
N_TRIALS     = 10

CB_LOG       = lgb.log_evaluation(100)

# ── 1. Load features + raw data ────────────────────────────────────────────────
print(f"\n[1] Loading features (d_num >= {FEAT_START})...")
t0 = time.time()
df = pd.read_parquet(
    FEAT_DIR,
    filters=[("d_num", ">=", FEAT_START)],
    columns=["id", "item_id", "cat_id", "dept_id", "store_id", "state_id",
             "d_num", "sales"] + NUM_FEATURES,
)
for col, dtype in CAT_DTYPES.items():
    if col in df.columns:
        df[col] = df[col].astype(dtype)

lag_cols = [f"lag_{l}" for l in [7, 14, 28, 56]]
df = df.dropna(subset=lag_cols).reset_index(drop=True)
print(f"    Loaded: {df.shape}  ({time.time()-t0:.1f}s)  "
      f"Memory: {df.memory_usage(deep=True).sum()/1e9:.2f} GB")

print("\n    Loading raw CSVs...")
t0 = time.time()
sales_eval  = pd.read_csv(os.path.join(DATA_RAW, "sales_train_evaluation.csv"))
prices_df   = pd.read_csv(os.path.join(DATA_RAW, "sell_prices.csv"))
calendar_df = pd.read_csv(os.path.join(DATA_RAW, "calendar.csv"))
print(f"    Loaded in {time.time()-t0:.1f}s")

# Pre-build lookups (used for recursive forecast)
cal_lookup   = build_calendar_lookup(calendar_df)
price_lookup = build_price_lookup(prices_df, calendar_df)

# Load forecast features from parquet (d_1914-1941, pre-computed single-step)
print("\n    Loading forecast features (d_1914-1941) from parquet...")
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

# Series ordering (sorted by id, used throughout)
series_meta = (
    df_fcast[["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]]
    .drop_duplicates("id")
    .sort_values("id")
    .reset_index(drop=True)
)
series_ids = series_meta["id"].values
n_series   = len(series_ids)
print(f"    Total series: {n_series:,}")

# ── helpers ───────────────────────────────────────────────────────────────────
actual_cols = [f"d_{d}" for d in range(LAST_TRAIN + 1, LAST_TRAIN + HORIZON + 1)]

def score_preds(preds_mat, sids):
    """Compute WRMSSE over full 30,490 series using actuals from sales_eval."""
    sub_sales = sales_eval[sales_eval["id"].isin(sids)].set_index("id").reindex(sids).reset_index()
    actuals   = sub_sales[actual_cols].values.astype(np.float32)
    score, _  = compute_wrmsse(
        preds=preds_mat, actuals=actuals,
        sales_df=sub_sales, prices_df=prices_df,
        calendar_df=calendar_df, last_train_day=LAST_TRAIN,
    )
    return float(score)

def predict_validation_single_step(model):
    """Single-step forecast using pre-computed parquet features (identical to Day 6)."""
    preds_flat = np.clip(model.predict(df_fcast[ALL_FEATURES]), 0, None).astype(np.float32)
    return preds_flat.reshape(n_series, HORIZON)

# ── 2. Load Day 6 global model ────────────────────────────────────────────────
print("\n[2] Loading Day 6 best global model...")
with open(os.path.join(MODELS, "lgbm_best.pkl"), "rb") as f:
    model_best = pickle.load(f)
print("    Loaded lgbm_best.pkl")

# ── 3. Sanity check: recursive on validation period ──────────────────────────
print(f"\n[3] Sanity check - recursive forecast on val period (d_{LAST_TRAIN+1}-d_{LAST_TRAIN+HORIZON})...")
buffer_val     = build_sales_buffer(sales_eval, series_ids, last_day=LAST_TRAIN)
val_price_by_d = build_eval_price_features(
    series_meta, price_lookup, cal_lookup,
    eval_start=LAST_TRAIN + 1, eval_end=LAST_TRAIN + HORIZON,
)
t0 = time.time()
preds_recursive_val = predict_recursive(
    model_best, buffer_val, series_meta, cal_lookup, val_price_by_d,
    CAT_DTYPES, start_day=LAST_TRAIN + 1, end_day=LAST_TRAIN + HORIZON,
)
wrmsse_recursive_val = score_preds(preds_recursive_val, series_ids)
wrmsse_singlestep    = score_preds(predict_validation_single_step(model_best), series_ids)
print(f"\n    Single-step WRMSSE: {wrmsse_singlestep:.4f}")
print(f"    Recursive    WRMSSE: {wrmsse_recursive_val:.4f}")
gap = abs(wrmsse_recursive_val - wrmsse_singlestep)
print(f"    Gap: {gap:.4f}  (expected ~5-15% for 28-step recursive over sparse M5 series)")

# ── 4. Per-category LightGBM — Optuna + retrain ───────────────────────────────
print("\n[4] Per-category LightGBM (FOODS / HOUSEHOLD / HOBBIES)...")

train_mask  = df["d_num"] <= VAL_START - 1
val_mask    = (df["d_num"] >= VAL_START) & (df["d_num"] <= LAST_TRAIN)
opt_t_mask  = (df["d_num"] >= OPTUNA_START) & (df["d_num"] <= VAL_START - 1)

cat_models  = {}
cat_scores  = {}
cat_results = {}

for cat in CATEGORIES:
    print(f"\n  -- {cat} --------------------------------------------------")
    cat_str_mask = df["cat_id"].astype(str) == cat

    X_tr = df.loc[train_mask & cat_str_mask, ALL_FEATURES]
    y_tr = df.loc[train_mask & cat_str_mask, "sales"].astype(np.float32)
    X_vl = df.loc[val_mask & cat_str_mask, ALL_FEATURES]
    y_vl = df.loc[val_mask & cat_str_mask, "sales"].astype(np.float32)
    X_op = df.loc[opt_t_mask & cat_str_mask, ALL_FEATURES]
    y_op = df.loc[opt_t_mask & cat_str_mask, "sales"].astype(np.float32)
    print(f"    Train: {len(X_tr):,}  Val: {len(X_vl):,}  Optuna subset: {len(X_op):,}")

    ds_tr  = lgb.Dataset(X_tr, label=y_tr, categorical_feature=CAT_COLS, free_raw_data=False)
    ds_vl  = lgb.Dataset(X_vl, label=y_vl, categorical_feature=CAT_COLS, reference=ds_tr, free_raw_data=False)
    ds_op  = lgb.Dataset(X_op, label=y_op, categorical_feature=CAT_COLS, free_raw_data=False)
    ds_opv = lgb.Dataset(X_vl, label=y_vl, categorical_feature=CAT_COLS, reference=ds_op, free_raw_data=False)

    # Optuna
    print(f"    Optuna ({N_TRIALS} trials)...")
    t_opt = time.time()

    def make_objective(ds_tr_, ds_vl_):
        def objective(trial):
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
                "feature_fraction":       trial.suggest_categorical("feat_frac",  [0.7, 0.8, 0.9]),
                "bagging_fraction":       trial.suggest_categorical("bag_frac",   [0.7, 0.8, 0.9]),
                "lambda_l2":              trial.suggest_categorical("l2",         [0.1, 0.5, 1.0]),
                "tweedie_variance_power": trial.suggest_float("tvp", 1.0, 1.9),
            }
            m = lgb.train(
                p, ds_tr_,
                num_boost_round=500,
                valid_sets=[ds_vl_],
                callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
            )
            return m.best_score["valid_0"]["tweedie"]
        return objective

    study = optuna.create_study(direction="minimize")
    study.optimize(make_objective(ds_op, ds_opv), n_trials=N_TRIALS, show_progress_bar=False)
    bp = study.best_params
    print(f"    Optuna done in {time.time()-t_opt:.0f}s  "
          f"Best val: {study.best_value:.4f}  "
          f"tvp={bp.get('tvp',1.5):.3f}  lr={bp.get('lr')}  leaves={bp.get('num_leaves')}")

    # Retrain on full data
    best_p = {
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
    print(f"    Retraining {cat} model...")
    t_train = time.time()
    model_cat = lgb.train(
        best_p, ds_tr,
        num_boost_round=3000,
        valid_sets=[ds_vl],
        callbacks=[lgb.early_stopping(75, verbose=False), CB_LOG],
    )
    print(f"    Best iter: {model_cat.best_iteration}  Time: {time.time()-t_train:.0f}s")

    # Validation score (single-step)
    cat_fcast_mask = df_fcast["cat_id"].astype(str) == cat
    cat_fcast_df   = df_fcast[cat_fcast_mask].sort_values(["id", "d_num"])
    cat_sids       = cat_fcast_df.groupby("id", sort=True).first().reset_index()["id"].values
    preds_cat_flat = np.clip(model_cat.predict(cat_fcast_df[ALL_FEATURES]), 0, None).astype(np.float32)
    preds_cat_mat  = preds_cat_flat.reshape(len(cat_sids), HORIZON)

    # Map to global series order for scoring
    global_cat_idx = np.searchsorted(series_ids, cat_sids)
    cat_val_full   = np.zeros((n_series, HORIZON), dtype=np.float32)
    cat_val_full[global_cat_idx] = preds_cat_mat
    # Score only within-category (pass just this cat's series + actuals)
    sub_sales = sales_eval[sales_eval["id"].isin(cat_sids)].set_index("id").reindex(cat_sids).reset_index()
    actuals_cat = sub_sales[actual_cols].values.astype(np.float32)
    cat_score, _ = compute_wrmsse(
        preds=preds_cat_mat, actuals=actuals_cat,
        sales_df=sub_sales, prices_df=prices_df,
        calendar_df=calendar_df, last_train_day=LAST_TRAIN,
    )
    print(f"    {cat} WRMSSE (val, single-step): {cat_score:.4f}")

    cat_models[cat] = model_cat
    cat_scores[cat] = {
        "wrmsse_val": float(cat_score),
        "best_iter": model_cat.best_iteration,
        "best_params": {k: float(v) if isinstance(v, float) else v for k, v in best_p.items()},
        "optuna_best_val": float(study.best_value),
    }

    with open(os.path.join(MODELS, f"lgbm_per_category_{cat}.pkl"), "wb") as f:
        pickle.dump(model_cat, f)
    print(f"    Saved lgbm_per_category_{cat}.pkl")

# ── 5. Build per-category full validation predictions (single-step) ───────────
print("\n[5] Building per-category validation predictions (single-step from parquet)...")
val_preds_per_cat = np.zeros((n_series, HORIZON), dtype=np.float32)
for cat in CATEGORIES:
    cat_fcast_mask = df_fcast["cat_id"].astype(str) == cat
    cat_fcast_df   = df_fcast[cat_fcast_mask].sort_values(["id", "d_num"])
    cat_sids       = cat_fcast_df.groupby("id", sort=True).first().reset_index()["id"].values
    preds_flat     = np.clip(cat_models[cat].predict(cat_fcast_df[ALL_FEATURES]), 0, None).astype(np.float32)
    preds_mat      = preds_flat.reshape(len(cat_sids), HORIZON)
    global_idx     = np.searchsorted(series_ids, cat_sids)
    val_preds_per_cat[global_idx] = preds_mat

wrmsse_per_cat_val = score_preds(val_preds_per_cat, series_ids)
print(f"    Per-category WRMSSE (val): {wrmsse_per_cat_val:.4f}")

# Blend validation predictions (single-step)
val_preds_global = predict_validation_single_step(model_best)
val_preds_blend  = 0.6 * val_preds_per_cat + 0.4 * val_preds_global
wrmsse_blend_val = score_preds(val_preds_blend, series_ids)
print(f"    Blend 0.6/0.4 WRMSSE (val): {wrmsse_blend_val:.4f}")
print(f"    Global single-step WRMSSE (val): {wrmsse_singlestep:.4f}")

# ── 6. Recursive evaluation period forecasts (d_1942–d_1969) ─────────────────
print(f"\n[6] Recursive eval forecast (d_{EVAL_START}-d_{EVAL_END})...")
eval_buffer    = build_sales_buffer(sales_eval, series_ids, last_day=LAST_TRAIN + HORIZON)
eval_price_by_d = build_eval_price_features(
    series_meta, price_lookup, cal_lookup, eval_start=EVAL_START, eval_end=EVAL_END,
)

# 6a. Global model recursive
print("\n  [6a] Global model eval recursive...")
t0 = time.time()
eval_preds_global = predict_recursive(
    model_best, eval_buffer, series_meta, cal_lookup, eval_price_by_d,
    CAT_DTYPES, start_day=EVAL_START, end_day=EVAL_END,
)
print(f"  Done in {time.time()-t0:.1f}s")

# 6b. Per-category model recursive (each category uses its own model)
print("\n  [6b] Per-category model eval recursive...")
eval_preds_per_cat = np.zeros((n_series, HORIZON), dtype=np.float32)
for cat in CATEGORIES:
    cat_mask    = series_meta["cat_id"].astype(str) == cat
    cat_indices = np.where(cat_mask)[0]
    cat_buffer  = eval_buffer[cat_indices]
    cat_meta    = series_meta[cat_mask].reset_index(drop=True)
    cat_price   = {d: eval_price_by_d[d][cat_indices] for d in eval_price_by_d}

    print(f"\n    {cat} ({len(cat_indices):,} series):")
    t0 = time.time()
    cat_eval_preds = predict_recursive(
        cat_models[cat], cat_buffer, cat_meta, cal_lookup, cat_price,
        CAT_DTYPES, start_day=EVAL_START, end_day=EVAL_END,
    )
    eval_preds_per_cat[cat_indices] = cat_eval_preds
    print(f"    {cat} done in {time.time()-t0:.1f}s")

# Blend eval predictions
eval_preds_blend = 0.6 * eval_preds_per_cat + 0.4 * eval_preds_global

# ── 7. Build Kaggle submissions ────────────────────────────────────────────────
print("\n[7] Building Kaggle submissions...")
sn28_base = pd.read_csv(os.path.join(SUBS, "seasonal_naive_28_submission.csv"))
f_cols    = [f"F{i}" for i in range(1, HORIZON + 1)]

def build_full_submission(val_mat, eval_mat, sids, fname):
    """
    Build 60,980-row submission:
      - _validation rows: F1..F28 = forecasts for d_1914–d_1941 (public LB)
      - _evaluation rows: F1..F28 = forecasts for d_1942–d_1969 (private LB)
    """
    base = sn28_base.copy().set_index("id")

    # Validation rows
    val_ids = [s.replace("_evaluation", "_validation") for s in sids]
    val_df  = pd.DataFrame(val_mat, columns=f_cols)
    val_df.insert(0, "id", val_ids)
    val_idx = pd.Index(val_ids).intersection(base.index)
    base.loc[val_idx] = val_df.set_index("id").loc[val_idx]

    # Evaluation rows
    eval_ids = list(sids)   # already have _evaluation suffix
    eval_df  = pd.DataFrame(eval_mat, columns=f_cols)
    eval_df.insert(0, "id", eval_ids)
    eval_idx = pd.Index(eval_ids).intersection(base.index)
    base.loc[eval_idx] = eval_df.set_index("id").loc[eval_idx]

    path = os.path.join(SUBS, fname)
    base.reset_index().to_csv(path, index=False)
    print(f"    Saved: {path}  ({len(val_idx)} val + {len(eval_idx)} eval rows)")
    return path

build_full_submission(val_preds_global,    eval_preds_global,    series_ids, "lgbm_global_recursive.csv")
build_full_submission(val_preds_per_cat,   eval_preds_per_cat,   series_ids, "lgbm_per_cat.csv")
build_full_submission(val_preds_blend,     eval_preds_blend,     series_ids, "lgbm_blend.csv")

# ── 8. Save results JSON ───────────────────────────────────────────────────────
results = {
    "sanity_check": {
        "single_step_wrmsse":  wrmsse_singlestep,
        "recursive_val_wrmsse": wrmsse_recursive_val,
        "gap": abs(wrmsse_recursive_val - wrmsse_singlestep),
    },
    "per_category": cat_scores,
    "full_val_wrmsse": {
        "global_single_step": wrmsse_singlestep,
        "per_cat_single_step": wrmsse_per_cat_val,
        "blend_0.6_0.4": wrmsse_blend_val,
    },
}

def make_serial(obj):
    if isinstance(obj, (np.floating, float)): return float(obj)
    if isinstance(obj, (np.integer, int)):    return int(obj)
    if isinstance(obj, dict):  return {k: make_serial(v) for k, v in obj.items()}
    if isinstance(obj, list):  return [make_serial(x) for x in obj]
    return obj

results_path = os.path.join(REPORTS, "day7_scores.json")
with open(results_path, "w") as f:
    json.dump(make_serial(results), f, indent=2)
print(f"\n    Saved scores: {results_path}")

# ── 9. Write per-category results report ──────────────────────────────────────
print("\n[9] Writing reports/06_per_category_models.md...")
report_lines = [
    "# Day 7 — Per-Category LightGBM + Recursive Evaluation Forecast",
    "",
    "## Overview",
    "",
    "Day 7 builds on Day 6 in two ways:",
    "1. **Recursive evaluation forecast** — generates predictions for the true evaluation",
    "   period d_1942–d_1969 (Kaggle private LB), fixing the Day 6 private score gap.",
    "2. **Per-category models** — separate LightGBM + Optuna for FOODS, HOUSEHOLD, HOBBIES,",
    "   exploiting the finding that optimal Tweedie variance power differs across demand regimes.",
    "",
    "---",
    "",
    "## Sanity Check: Recursive vs Single-Step (Validation Period)",
    "",
    "| Forecast method | WRMSSE (val d_1914–1941) |",
    "|-----------------|------------------------|",
    f"| Global single-step (Day 6) | {wrmsse_singlestep:.4f} |",
    f"| Global recursive | {wrmsse_recursive_val:.4f} |",
    f"| Gap | {abs(wrmsse_recursive_val - wrmsse_singlestep):.4f} |",
    "",
    "Single-step uses pre-computed features from actual sales — strictly better than recursive.",
    "The gap quantifies error propagation over 28 recursive steps.",
    "",
    "---",
    "",
    "## Per-Category Optuna Results",
    "",
    "| Category | tvp | lr | num_leaves | Val WRMSSE | vs Global (0.5422) |",
    "|----------|-----|----|------------|-----------|-------------------|",
]
for cat in CATEGORIES:
    s = cat_scores[cat]
    p = s["best_params"]
    report_lines.append(
        f"| {cat} | {p.get('tweedie_variance_power', 0):.3f} | "
        f"{p.get('learning_rate', 0)} | {p.get('num_leaves', 0)} | "
        f"{s['wrmsse_val']:.4f} | "
        f"{'−' if s['wrmsse_val'] < 0.5422 else '+'}{abs(s['wrmsse_val'] - 0.5422):.4f} |"
    )

report_lines += [
    "",
    "**Hypothesis vs reality:** Did the optimal tvp differ across categories?",
    "- FOODS (dense demand): lower tvp expected (≈ 1.0–1.3)",
    "- HOUSEHOLD (moderate intermittency): medium tvp expected (≈ 1.2–1.6)",
    "- HOBBIES (high intermittency): higher tvp expected (≈ 1.5–1.9)",
    "",
    "---",
    "",
    "## Validation Period Comparison (d_1914–1941)",
    "",
    "| Model | WRMSSE (full 30,490) | vs Day 6 global |",
    "|-------|---------------------|-----------------|",
    f"| Global LightGBM (Day 6) | {wrmsse_singlestep:.4f} | baseline |",
    f"| Per-category LightGBM | {wrmsse_per_cat_val:.4f} | "
    f"{'−' if wrmsse_per_cat_val < wrmsse_singlestep else '+'}{abs(wrmsse_per_cat_val - wrmsse_singlestep):.4f} |",
    f"| Blend (0.6×per_cat + 0.4×global) | {wrmsse_blend_val:.4f} | "
    f"{'−' if wrmsse_blend_val < wrmsse_singlestep else '+'}{abs(wrmsse_blend_val - wrmsse_singlestep):.4f} |",
    "",
    "---",
    "",
    "## Kaggle Submission Files",
    "",
    "| File | Val source | Eval source | Note |",
    "|------|-----------|------------|------|",
    "| lgbm_global_recursive.csv | Global single-step | Global recursive | Day 6 model, now with real eval forecast |",
    "| lgbm_per_cat.csv | Per-cat single-step | Per-cat recursive | Category-specific models |",
    "| lgbm_blend.csv | 0.6×per_cat + 0.4×global | Same blend | Best expected private LB |",
    "",
    "Kaggle scores: TBD after submission.",
    "",
    "---",
    "",
    "## Why Per-Category?",
    "",
    "The global model optimises a single Tweedie variance power across all categories.",
    "The M5 dataset has three structurally different demand regimes:",
    "- **FOODS**: high volume, regular, low zero-rate (62%) → low tvp (near-Gaussian)",
    "- **HOUSEHOLD**: medium volume, moderate intermittency → medium tvp",
    "- **HOBBIES**: low volume, high intermittency, 77% zeros → high tvp (Poisson-like)",
    "",
    "Per-category training lets each model find its optimal tvp independently.",
    "",
    "---",
    "",
    "## Files",
    "",
    "| File | Description |",
    "|------|-------------|",
    "| `scripts/07_train_per_category.py` | Full training + recursive forecast pipeline |",
    "| `src/models/recursive_forecast.py` | Recursive forecast library |",
    "| `data/models/lgbm_per_category_{FOODS,HOUSEHOLD,HOBBIES}.pkl` | Trained models (gitignored) |",
    "| `reports/day7_scores.json` | All scores and parameters |",
    "| `submissions/lgbm_global_recursive.csv` | Global model, full eval period |",
    "| `submissions/lgbm_per_cat.csv` | Per-category models |",
    "| `submissions/lgbm_blend.csv` | Blend submission |",
]

report_path = os.path.join(REPORTS, "06_per_category_models.md")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines) + "\n")
print(f"    Saved: {report_path}")

# ── summary ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("DAY 7 SUMMARY")
print("=" * 60)
print(f"\nSanity check (recursive vs single-step):")
print(f"  Single-step: {wrmsse_singlestep:.4f}  Recursive: {wrmsse_recursive_val:.4f}")
print(f"\nPer-category val WRMSSE (single-step):")
for cat in CATEGORIES:
    tvp = cat_scores[cat]["best_params"].get("tweedie_variance_power", 0)
    print(f"  {cat}: {cat_scores[cat]['wrmsse_val']:.4f}  (tvp={tvp:.3f})")
print(f"\nFull-catalogue val WRMSSE:")
print(f"  Global (Day 6):       {wrmsse_singlestep:.4f}")
print(f"  Per-category:         {wrmsse_per_cat_val:.4f}")
print(f"  Blend (0.6/0.4):      {wrmsse_blend_val:.4f}")
print(f"\nSubmissions built:")
print(f"  submissions/lgbm_global_recursive.csv")
print(f"  submissions/lgbm_per_cat.csv")
print(f"  submissions/lgbm_blend.csv")
print("\nDone! Submit lgbm_blend.csv to Kaggle for private LB score.")
