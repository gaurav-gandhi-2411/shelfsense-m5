"""
Run a single classical method on the 1k sample and save results.
Usage: python run_one_method.py <method>  (ets|arima|sarima|sarimax)
"""
from __future__ import annotations

import sys
import os
import time
import json
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_ROOT)

MAIN_REPO = PROJ_ROOT
DATA_RAW = os.path.join(MAIN_REPO, "data", "raw", "m5-forecasting-accuracy")
DATA_PROCESSED = os.path.join(MAIN_REPO, "data", "processed")
REPORTS_DIR = os.path.join(PROJ_ROOT, "reports")
SUBMISSIONS_DIR = os.path.join(MAIN_REPO, "submissions")
RESULTS_DIR = os.path.join(MAIN_REPO, "data", "processed", "day3_results")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(SUBMISSIONS_DIR, exist_ok=True)

method = sys.argv[1] if len(sys.argv) > 1 else "ets"
print(f"Running method: {method.upper()}")

# Check if already done
out_preds = os.path.join(RESULTS_DIR, f"{method}_preds.npy")
out_meta = os.path.join(RESULTS_DIR, f"{method}_meta.json")
if os.path.exists(out_preds) and os.path.exists(out_meta):
    print(f"Already done: {out_preds}")
    sys.exit(0)

print("Loading data...")
t0 = time.time()
sales_eval = pd.read_csv(os.path.join(DATA_RAW, "sales_train_evaluation.csv"))
calendar_df = pd.read_csv(os.path.join(DATA_RAW, "calendar.csv"))
prices_df = pd.read_csv(os.path.join(DATA_RAW, "sell_prices.csv"))
print(f"  Loaded in {time.time()-t0:.1f}s")

LAST_TRAIN_DAY = 1913
HORIZON = 28
train_cols = [f"d_{d}" for d in range(1, LAST_TRAIN_DAY + 1)]
actual_cols = [f"d_{d}" for d in range(LAST_TRAIN_DAY + 1, LAST_TRAIN_DAY + HORIZON + 1)]

# Load sample IDs
sample_csv = os.path.join(REPORTS_DIR, "sample_1000_series.csv")
if not os.path.exists(sample_csv):
    sample_csv = os.path.join(DATA_PROCESSED, "sample_1000_series.csv")
sample_df = pd.read_csv(sample_csv)
sample_ids = sample_df["id"].tolist()
print(f"  Sample: {len(sample_ids)} series")

sales_sub = sales_eval[sales_eval["id"].isin(sample_ids)].copy()
sales_sub = sales_sub.set_index("id").reindex(sample_ids).reset_index()
actuals_sub = sales_sub[actual_cols].values.astype(np.float32)

from models.classical import run_batch
from evaluation.wrmsse import compute_wrmsse

print(f"Running {method.upper()} with n_jobs=4...")
t_start = time.time()
preds, meta = run_batch(
    method=method,
    sample_ids=sample_ids,
    sales_train=sales_eval,
    prices_df=prices_df,
    calendar_df=calendar_df,
    n_jobs=4,
    horizon=HORIZON,
    last_train_day=LAST_TRAIN_DAY,
)
wall_time = time.time() - t_start
meta["wall_time_seconds"] = wall_time
meta["wall_time_per_series"] = wall_time / len(sample_ids)
print(f"  Done in {wall_time:.1f}s ({wall_time/len(sample_ids):.2f}s/series)")

# Score
score, level_scores = compute_wrmsse(
    preds=preds.astype(np.float32),
    actuals=actuals_sub,
    sales_df=sales_sub,
    prices_df=prices_df,
    calendar_df=calendar_df,
    last_train_day=LAST_TRAIN_DAY,
)
meta["wrmsse_sample"] = float(score)
meta["wrmsse_by_level"] = {k: float(v) for k, v in level_scores.items()}

# Category-level
def compute_category_wrmsse(preds, actuals, sales_subset):
    results = {}
    for cat in ["FOODS", "HOUSEHOLD", "HOBBIES"]:
        mask = sales_subset["cat_id"] == cat
        if mask.sum() == 0:
            continue
        idx = np.where(mask.values)[0]
        sub_sales = sales_subset[mask].reset_index(drop=True)
        sub_preds = preds[idx]
        sub_actuals = actuals[idx]
        s, _ = compute_wrmsse(
            preds=sub_preds.astype(np.float32),
            actuals=sub_actuals.astype(np.float32),
            sales_df=sub_sales,
            prices_df=prices_df,
            calendar_df=calendar_df,
            last_train_day=LAST_TRAIN_DAY,
        )
        results[cat] = float(s)
    return results

cat_scores = compute_category_wrmsse(preds, actuals_sub, sales_sub)
meta["wrmsse_by_category"] = cat_scores
print(f"  WRMSSE: {score:.4f}")
print(f"  By cat: {cat_scores}")
print(f"  Fallbacks: {meta['n_fallbacks']}")

# Save
np.save(out_preds, preds)
with open(out_meta, "w") as f:
    json.dump(meta, f, indent=2)
print(f"  Saved: {out_preds}")

# Submission CSV
HORIZON_COLS = [f"F{i}" for i in range(1, HORIZON + 1)]
sub_df = pd.DataFrame(preds, columns=HORIZON_COLS)
sub_df.insert(0, "id", sample_ids)
sub_path = os.path.join(SUBMISSIONS_DIR, f"{method}_sample_submission.csv")
sub_df.to_csv(sub_path, index=False)
print(f"  Submission: {sub_path}")
print("Done!")
