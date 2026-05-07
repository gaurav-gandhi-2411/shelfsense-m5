"""
Day 3: Classical Statistical Methods
Full pipeline: sample selection, model fitting, scoring, and report generation.
"""
from __future__ import annotations

import sys
import os
import time
import json
import warnings

warnings.filterwarnings("ignore")

# Ensure anaconda site-packages are on path
_SP = "C:/Users/gaura/anaconda3/Lib/site-packages"
if _SP not in sys.path:
    sys.path.insert(0, _SP)

# Project root on path
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_ROOT)

import numpy as np
import pandas as pd

print("=" * 60)
print("DAY 3 — Classical Forecasting Methods")
print("=" * 60)

# ── paths ──────────────────────────────────────────────────────────────────────
# Data lives in the main repo (not the worktree), reports/submissions in worktree
MAIN_REPO = "C:/Users/gaura/ml-projects/shelfsense-m5"
DATA_RAW = os.path.join(MAIN_REPO, "data", "raw", "m5-forecasting-accuracy")
DATA_PROCESSED = os.path.join(MAIN_REPO, "data", "processed")
REPORTS_DIR = os.path.join(PROJ_ROOT, "reports")
SUBMISSIONS_DIR = os.path.join(MAIN_REPO, "submissions")

os.makedirs(DATA_PROCESSED, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(SUBMISSIONS_DIR, exist_ok=True)

# ── load data ──────────────────────────────────────────────────────────────────
print("\n[1] Loading M5 data...")
t_load = time.time()

sales_eval = pd.read_csv(os.path.join(DATA_RAW, "sales_train_evaluation.csv"))
calendar_df = pd.read_csv(os.path.join(DATA_RAW, "calendar.csv"))
prices_df = pd.read_csv(os.path.join(DATA_RAW, "sell_prices.csv"))

print(f"    sales_train_evaluation: {sales_eval.shape}")
print(f"    calendar: {calendar_df.shape}")
print(f"    prices: {prices_df.shape}")
print(f"    Load time: {time.time()-t_load:.1f}s")

LAST_TRAIN_DAY = 1913
HORIZON = 28

train_cols = [f"d_{d}" for d in range(1, LAST_TRAIN_DAY + 1)]
actual_cols = [f"d_{d}" for d in range(LAST_TRAIN_DAY + 1, LAST_TRAIN_DAY + HORIZON + 1)]

META_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: Sample selection
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[2] Selecting stratified 1,000-series sample...")

# Compute total training sales (d_1 to d_1913)
sales_meta = sales_eval[META_COLS].copy()
sales_meta["total_sales"] = sales_eval[train_cols].sum(axis=1).values

# ── FOODS: top 334 by total sales ─────────────────────────────────────────────
foods = sales_meta[sales_meta["cat_id"] == "FOODS"].copy()
foods_sorted = foods.sort_values("total_sales", ascending=False)
foods_sample = foods_sorted.head(334).copy()
foods_sample["stratum"] = "FOODS_top"

print(f"    FOODS candidates: {len(foods)}, sampled: {len(foods_sample)}")
print(f"    FOODS top sales: {foods_sample['total_sales'].min():.0f} – {foods_sample['total_sales'].max():.0f}")

# ── HOUSEHOLD: 333 closest to median sales rank ───────────────────────────────
hh = sales_meta[sales_meta["cat_id"] == "HOUSEHOLD"].copy()
hh = hh.sort_values("total_sales")
hh["rank"] = np.arange(len(hh))
median_rank = len(hh) // 2
half = 333 // 2
start_idx = max(0, median_rank - half)
end_idx = start_idx + 333
if end_idx > len(hh):
    end_idx = len(hh)
    start_idx = end_idx - 333
hh_sample = hh.iloc[start_idx:end_idx].copy()
hh_sample = hh_sample.drop(columns=["rank"])
hh_sample["stratum"] = "HOUSEHOLD_mid"

print(f"    HOUSEHOLD candidates: {len(hh)}, sampled: {len(hh_sample)}")
print(f"    HOUSEHOLD sales range: {hh_sample['total_sales'].min():.0f} – {hh_sample['total_sales'].max():.0f}")

# ── HOBBIES: 333 lowest sellers ───────────────────────────────────────────────
hob = sales_meta[sales_meta["cat_id"] == "HOBBIES"].copy()
hob_sorted = hob.sort_values("total_sales", ascending=True)
hob_sample = hob_sorted.head(333).copy()
hob_sample["stratum"] = "HOBBIES_low"

print(f"    HOBBIES candidates: {len(hob)}, sampled: {len(hob_sample)}")
print(f"    HOBBIES low sales: {hob_sample['total_sales'].min():.0f} – {hob_sample['total_sales'].max():.0f}")

# ── combine ────────────────────────────────────────────────────────────────────
sample_df = pd.concat(
    [foods_sample[META_COLS + ["total_sales", "stratum"]],
     hh_sample[META_COLS + ["total_sales", "stratum"]],
     hob_sample[META_COLS + ["total_sales", "stratum"]]],
    ignore_index=True
)
assert len(sample_df) == 1000, f"Expected 1000 series, got {len(sample_df)}"

print(f"    Total sampled: {len(sample_df)}")
print(f"    By stratum: {sample_df['stratum'].value_counts().to_dict()}")

# Save to both locations (data/processed + reports for git)
sample_csv_data = os.path.join(DATA_PROCESSED, "sample_1000_series.csv")
sample_csv_reports = os.path.join(REPORTS_DIR, "sample_1000_series.csv")
sample_df.to_csv(sample_csv_data, index=False)
sample_df.to_csv(sample_csv_reports, index=False)
print(f"    Saved to {sample_csv_data}")
print(f"    Saved to {sample_csv_reports}")

sample_ids = sample_df["id"].tolist()

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: Prepare subset data for modeling and scoring
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[3] Preparing subset data...")

sales_sub = sales_eval[sales_eval["id"].isin(sample_ids)].copy()
sales_sub = sales_sub.set_index("id").reindex(sample_ids).reset_index()
assert len(sales_sub) == 1000

actuals_sub = sales_sub[actual_cols].values.astype(np.float32)
train_mat_sub = sales_sub[train_cols].values.astype(np.float32)

print(f"    Subset shape: {sales_sub.shape}")
print(f"    Actuals shape: {actuals_sub.shape}")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: Seasonal Naive 28 reference on same 1k sample
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[4] Computing Seasonal Naive 28 reference on 1k sample...")

sys.path.insert(0, os.path.join(PROJ_ROOT, "src"))
from models.naive import seasonal_naive
from evaluation.wrmsse import compute_wrmsse

sn28_preds = seasonal_naive(train_mat_sub, period=28, horizon=28)
sn28_score, sn28_levels = compute_wrmsse(
    preds=sn28_preds.astype(np.float32),
    actuals=actuals_sub,
    sales_df=sales_sub,
    prices_df=prices_df,
    calendar_df=calendar_df,
    last_train_day=LAST_TRAIN_DAY,
)
print(f"    Seasonal Naive 28 (1k sample): WRMSSE = {sn28_score:.4f}")

# Category-level WRMSSE for seasonal naive reference
def compute_category_wrmsse(preds, actuals, sales_subset):
    """Compute WRMSSE per category by filtering subset further."""
    results = {}
    for cat in ["FOODS", "HOUSEHOLD", "HOBBIES"]:
        mask = sales_subset["cat_id"] == cat
        if mask.sum() == 0:
            continue
        idx = np.where(mask.values)[0]
        sub_sales = sales_subset[mask].reset_index(drop=True)
        sub_preds = preds[idx]
        sub_actuals = actuals[idx]
        score, _ = compute_wrmsse(
            preds=sub_preds.astype(np.float32),
            actuals=sub_actuals.astype(np.float32),
            sales_df=sub_sales,
            prices_df=prices_df,
            calendar_df=calendar_df,
            last_train_day=LAST_TRAIN_DAY,
        )
        results[cat] = score
    return results

sn28_cat = compute_category_wrmsse(sn28_preds.astype(np.float32), actuals_sub, sales_sub)
print(f"    SN28 by category: {sn28_cat}")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4: Run classical methods
# ═══════════════════════════════════════════════════════════════════════════════
from models.classical import run_batch

all_results = {}
METHODS = ["ets", "arima", "sarima", "sarimax"]
N_JOBS = 4

print(f"\n[5] Running classical methods with n_jobs={N_JOBS}...")
print("    Expected time: ETS ~3min, ARIMA ~20min, SARIMA ~35min, SARIMAX ~50min")

for method in METHODS:
    print(f"\n  --- {method.upper()} ---")
    t_start = time.time()

    preds, meta = run_batch(
        method=method,
        sample_ids=sample_ids,
        sales_train=sales_eval,
        prices_df=prices_df,
        calendar_df=calendar_df,
        n_jobs=N_JOBS,
        horizon=HORIZON,
        last_train_day=LAST_TRAIN_DAY,
    )

    wall_time = time.time() - t_start
    meta["wall_time_seconds"] = wall_time
    meta["wall_time_per_series"] = wall_time / len(sample_ids)

    # Score
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

    # Category-level
    cat_scores = compute_category_wrmsse(preds, actuals_sub, sales_sub)
    meta["wrmsse_by_category"] = cat_scores

    print(f"    WRMSSE (1k sample): {score:.4f}")
    print(f"    By category: {cat_scores}")
    print(f"    Wall time: {wall_time:.1f}s ({wall_time/len(sample_ids):.2f}s/series)")
    print(f"    Fallbacks: {meta['n_fallbacks']}, Zero forecasts: {meta['n_zero_forecasts']}")

    all_results[method] = meta

    # Save submission CSV
    sub_df = pd.DataFrame(
        preds,
        columns=[f"F{i}" for i in range(1, HORIZON + 1)]
    )
    sub_df.insert(0, "id", sample_ids)
    sub_path = os.path.join(SUBMISSIONS_DIR, f"{method}_sample_submission.csv")
    sub_df.to_csv(sub_path, index=False)
    print(f"    Saved: {sub_path}")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5: Save results JSON
# ═══════════════════════════════════════════════════════════════════════════════
results_path = os.path.join(REPORTS_DIR, "day3_classical_scores.json")

def make_serializable(obj):
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_serializable(x) for x in obj]
    return obj

results_to_save = {
    "seasonal_naive_28_reference": {
        "wrmsse_sample": float(sn28_score),
        "wrmsse_by_category": {k: float(v) for k, v in sn28_cat.items()},
        "wrmsse_by_level": {k: float(v) for k, v in sn28_levels.items()},
    },
    "methods": make_serializable(all_results),
}

with open(results_path, "w") as f:
    json.dump(results_to_save, f, indent=2)
print(f"\n[6] Saved scores to {results_path}")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6: Print final summary
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("FINAL RESULTS SUMMARY")
print("=" * 60)

print(f"\nReference: Seasonal Naive 28 on 1k sample = {sn28_score:.4f}")
print(f"  FOODS: {sn28_cat.get('FOODS', 'N/A'):.4f}  "
      f"HOUSEHOLD: {sn28_cat.get('HOUSEHOLD', 'N/A'):.4f}  "
      f"HOBBIES: {sn28_cat.get('HOBBIES', 'N/A'):.4f}")

print("\nClassical Methods:")
for method in METHODS:
    r = all_results[method]
    s = r['wrmsse_sample']
    cat = r.get('wrmsse_by_category', {})
    t = r['wall_time_seconds']
    fb = r['n_fallbacks']
    print(f"\n  {method.upper()}: WRMSSE={s:.4f} (vs SN28 ref {sn28_score:.4f})")
    print(f"    FOODS:{cat.get('FOODS','?'):.4f}  HOUSEHOLD:{cat.get('HOUSEHOLD','?'):.4f}  HOBBIES:{cat.get('HOBBIES','?'):.4f}")
    print(f"    Time: {t:.1f}s total ({t/1000:.2f}s/series), Fallbacks: {fb}")

print("\nDone!")
