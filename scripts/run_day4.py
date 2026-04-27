"""
Day 4: Prophet model + hyperparameter sweep on changepoint_prior_scale.
Runs on the same 1,000-series stratified sample as Day 3.
"""
from __future__ import annotations

import sys
import os
import time
import json
import warnings

warnings.filterwarnings("ignore")

_SP = "C:/Users/gaura/anaconda3/Lib/site-packages"
if _SP not in sys.path:
    sys.path.insert(0, _SP)

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_ROOT)

import numpy as np
import pandas as pd

print("=" * 60)
print("DAY 4 — Prophet Forecasting")
print("=" * 60)

# ── paths ──────────────────────────────────────────────────────────────────────
MAIN_REPO = "C:/Users/gaura/ml-projects/shelfsense-m5"
DATA_RAW = os.path.join(MAIN_REPO, "data", "raw", "m5-forecasting-accuracy")
DATA_PROCESSED = os.path.join(MAIN_REPO, "data", "processed")
REPORTS_DIR = os.path.join(PROJ_ROOT, "reports")
SUBMISSIONS_DIR = os.path.join(MAIN_REPO, "submissions")

# ── load data ──────────────────────────────────────────────────────────────────
print("\n[1] Loading M5 data...")
t0 = time.time()
sales_eval = pd.read_csv(os.path.join(DATA_RAW, "sales_train_evaluation.csv"))
calendar_df = pd.read_csv(os.path.join(DATA_RAW, "calendar.csv"))
prices_df = pd.read_csv(os.path.join(DATA_RAW, "sell_prices.csv"))
print(f"    Loaded in {time.time()-t0:.1f}s")

LAST_TRAIN_DAY = 1913
HORIZON = 28
train_cols = [f"d_{d}" for d in range(1, LAST_TRAIN_DAY + 1)]
actual_cols = [f"d_{d}" for d in range(LAST_TRAIN_DAY + 1, LAST_TRAIN_DAY + HORIZON + 1)]
META_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]

# ── load sample ────────────────────────────────────────────────────────────────
print("\n[2] Loading 1k-series sample from Day 3...")
sample_csv = os.path.join(REPORTS_DIR, "sample_1000_series.csv")
sample_df = pd.read_csv(sample_csv)
sample_ids = sample_df["id"].tolist()
print(f"    {len(sample_ids)} series loaded")

sales_sub = (
    sales_eval[sales_eval["id"].isin(sample_ids)]
    .set_index("id")
    .reindex(sample_ids)
    .reset_index()
)
actuals_sub = sales_sub[actual_cols].values.astype(np.float32)
train_mat_sub = sales_sub[train_cols].values.astype(np.float32)

# ── WRMSSE evaluator ───────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(PROJ_ROOT, "src"))
from evaluation.wrmsse import compute_wrmsse
from models.prophet_model import run_batch, build_m5_holidays

# Reference: SN28 on 1k sample (computed Day 3)
SN28_REFERENCE = 0.6778

print(f"\n    Reference: Seasonal Naive 28 (1k sample) = {SN28_REFERENCE:.4f}")

# ── changepoint sweep ──────────────────────────────────────────────────────────
CPS_VALUES = [0.01, 0.05, 0.1]
N_JOBS = 2   # conservative; Prophet Stan backend is memory-heavy

sweep_results = {}

print(f"\n[3] Prophet changepoint_prior_scale sweep: {CPS_VALUES}")
print(f"    n_jobs={N_JOBS}  (conservative to avoid OOM)")

best_cps = None
best_score = float("inf")
best_preds = None

for cps in CPS_VALUES:
    print(f"\n  --- Prophet (cps={cps}) ---")
    t_start = time.time()

    preds, meta = run_batch(
        sample_ids=sample_ids,
        sales_train=sales_eval,
        prices_df=prices_df,
        calendar_df=calendar_df,
        changepoint_prior_scale=cps,
        n_jobs=N_JOBS,
        horizon=HORIZON,
        last_train_day=LAST_TRAIN_DAY,
    )

    wall = time.time() - t_start
    meta["wall_time_seconds"] = wall

    score, level_scores = compute_wrmsse(
        preds=preds.astype(np.float32),
        actuals=actuals_sub,
        sales_df=sales_sub,
        prices_df=prices_df,
        calendar_df=calendar_df,
        last_train_day=LAST_TRAIN_DAY,
    )
    meta["wrmsse_sample"] = score
    meta["wrmsse_by_level"] = level_scores

    # Per-category
    cat_scores = {}
    for cat in ["FOODS", "HOUSEHOLD", "HOBBIES"]:
        mask = sales_sub["cat_id"] == cat
        idx = np.where(mask.values)[0]
        sub_sales = sales_sub[mask].reset_index(drop=True)
        s, _ = compute_wrmsse(
            preds=preds[idx].astype(np.float32),
            actuals=actuals_sub[idx].astype(np.float32),
            sales_df=sub_sales,
            prices_df=prices_df,
            calendar_df=calendar_df,
            last_train_day=LAST_TRAIN_DAY,
        )
        cat_scores[cat] = s
    meta["wrmsse_by_category"] = cat_scores

    print(f"    WRMSSE (1k sample): {score:.4f}  (vs SN28 ref {SN28_REFERENCE:.4f})")
    print(f"    FOODS: {cat_scores['FOODS']:.4f}  HOUSEHOLD: {cat_scores['HOUSEHOLD']:.4f}  HOBBIES: {cat_scores['HOBBIES']:.4f}")
    print(f"    Wall time: {wall:.1f}s ({wall/len(sample_ids):.2f}s/series)")
    print(f"    Fallbacks: {meta['n_fallbacks']}, Zero forecasts: {meta['n_zero_forecasts']}")

    sweep_results[str(cps)] = meta

    if score < best_score:
        best_score = score
        best_cps = cps
        best_preds = preds.copy()

print(f"\n  Best cps: {best_cps}  ->  WRMSSE = {best_score:.4f}")

# ── save best Prophet submission ───────────────────────────────────────────────
print(f"\n[4] Saving best Prophet submission (cps={best_cps})...")
sub_df = pd.DataFrame(
    best_preds,
    columns=[f"F{i}" for i in range(1, HORIZON + 1)],
)
sub_df.insert(0, "id", sample_ids)
sub_path = os.path.join(SUBMISSIONS_DIR, "prophet_sample_submission.csv")
sub_df.to_csv(sub_path, index=False)
print(f"    Saved: {sub_path}")

# ── save scores JSON ───────────────────────────────────────────────────────────
def make_serializable(obj):
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_serializable(x) for x in obj]
    return obj

scores_path = os.path.join(REPORTS_DIR, "day4_prophet_scores.json")
with open(scores_path, "w") as f:
    json.dump(make_serializable({"sweep": sweep_results, "best_cps": best_cps}), f, indent=2)
print(f"    Saved scores: {scores_path}")

# ── summary ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("DAY 4 SUMMARY — Prophet Results")
print("=" * 60)
print(f"\n{'cps':>6}  {'WRMSSE':>8}  {'vs ref':>8}")
print("-" * 28)
for cps_key, r in sweep_results.items():
    delta = r["wrmsse_sample"] - SN28_REFERENCE
    sign = "+" if delta > 0 else ""
    print(f"{cps_key:>6}  {r['wrmsse_sample']:>8.4f}  {sign}{delta:>7.4f}")

print(f"\nBest Prophet (cps={best_cps}): WRMSSE = {best_score:.4f}")
print(f"SN28 reference (1k):           WRMSSE = {SN28_REFERENCE:.4f}")
print(f"ETS (1k):                       WRMSSE = 0.6541")
print(f"ARIMA (1k):                     WRMSSE = 0.7493")
print("\nDone!")
