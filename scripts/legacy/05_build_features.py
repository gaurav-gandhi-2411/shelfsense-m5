"""
Day 5: Build the full M5 feature dataset and write to data/processed/features/.

Output: 10 parquet files (one per store), readable together with pd.read_parquet().
"""
from __future__ import annotations

import sys
import os
import time
import random
import warnings

warnings.filterwarnings("ignore")

_SP = "C:/Users/gaura/anaconda3/Lib/site-packages"
if _SP not in sys.path:
    sys.path.insert(0, _SP)

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_ROOT)
sys.path.insert(0, os.path.join(PROJ_ROOT, "src"))

import numpy as np
import pandas as pd

print("=" * 60)
print("DAY 5 — Feature Engineering Pipeline")
print("=" * 60)

DATA_RAW  = os.path.join(PROJ_ROOT, "data", "raw", "m5-forecasting-accuracy")
DATA_PROC = os.path.join(PROJ_ROOT, "data", "processed")
FEAT_DIR  = os.path.join(DATA_PROC, "features")
REPORTS   = os.path.join(PROJ_ROOT, "reports")
os.makedirs(FEAT_DIR, exist_ok=True)

# ── load raw data ──────────────────────────────────────────────────────────────
print("\n[1] Loading M5 raw data...")
t0 = time.time()
sales_eval  = pd.read_csv(os.path.join(DATA_RAW, "sales_train_evaluation.csv"))
calendar_df = pd.read_csv(os.path.join(DATA_RAW, "calendar.csv"))
prices_df   = pd.read_csv(os.path.join(DATA_RAW, "sell_prices.csv"))
print(f"    sales: {sales_eval.shape}  calendar: {calendar_df.shape}  prices: {prices_df.shape}")
print(f"    Load time: {time.time()-t0:.1f}s")

# ── run pipeline ──────────────────────────────────────────────────────────────
print("\n[2] Running feature pipeline (batch by store)...")
from features.pipeline import feature_engineer

total_rows = feature_engineer(
    sales_df=sales_eval,
    calendar_df=calendar_df,
    prices_df=prices_df,
    output_dir=FEAT_DIR,
    last_day=1941,
    verbose=True,
)

# ── sanity checks ──────────────────────────────────────────────────────────────
print("\n[3] Running sanity checks...")

# Load full feature dataset
print("    Loading all parquet files...", flush=True)
df_all = pd.read_parquet(FEAT_DIR)
print(f"    Shape: {df_all.shape}")

EXPECTED_ROWS = 30490 * 1941
print(f"\n  [CHECK 1] Row count: {len(df_all):,} (expected {EXPECTED_ROWS:,})", end="")
if len(df_all) == EXPECTED_ROWS:
    print(" OK")
else:
    print(f" FAIL  DIFF = {len(df_all) - EXPECTED_ROWS:+,}")

# Check categorical dtypes
print("\n  [CHECK 2] Categorical dtypes:")
cat_cols = ["cat_id", "dept_id", "store_id", "state_id"]
for col in cat_cols:
    is_cat = str(df_all[col].dtype) == "category"
    print(f"    {col}: {df_all[col].dtype}  {'OK' if is_cat else 'FAIL'}")

# Check NaN in lag/rolling after day 180
print("\n  [CHECK 3] NaN in lag/rolling features after d_num > 180:")
lag_roll_cols = [c for c in df_all.columns if c.startswith("lag_") or c.startswith("roll_")]
late = df_all[df_all["d_num"] > 180]
nan_counts = late[lag_roll_cols].isna().sum()
nan_nonzero = nan_counts[nan_counts > 0]
if len(nan_nonzero) == 0:
    print("    No NaN found OK")
else:
    print(f"    NaN found in: {nan_nonzero.to_dict()} ← expected for std with 1 obs")

# Spot-check lag_7: at d_num=100, lag_7 should equal sales at d_num=93
print("\n  [CHECK 4] Spot-check lag_7 correctness (d_num=100 vs d_num=93):")
random.seed(42)
check_ids = random.sample(df_all["id"].unique().tolist(), 3)
all_ok = True
for sid in check_ids:
    s = df_all[df_all["id"] == sid].sort_values("d_num")
    v93  = s.loc[s["d_num"] == 93,  "sales"].values[0]
    v100 = s.loc[s["d_num"] == 100, "lag_7"].values[0]
    ok = abs(float(v100) - float(v93)) < 1e-4
    print(f"    {sid}: sales[93]={v93:.2f}  lag_7[100]={v100:.2f}  {'OK' if ok else 'FAIL'}")
    if not ok:
        all_ok = False

# ── disk size summary ─────────────────────────────────────────────────────────
print("\n[4] Disk size summary:")
parquet_files = [f for f in os.listdir(FEAT_DIR) if f.endswith(".parquet")]
total_mb = 0.0
for f in sorted(parquet_files):
    mb = os.path.getsize(os.path.join(FEAT_DIR, f)) / 1e6
    total_mb += mb
    print(f"    {f}: {mb:.1f} MB")
print(f"    Total: {total_mb:.1f} MB ({total_mb/1024:.2f} GB)")

# ── feature list ──────────────────────────────────────────────────────────────
print("\n[5] Feature columns:")
feature_cols = [c for c in df_all.columns if c not in
                ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id", "d", "d_num", "sales"]]
print(f"    Total features: {len(feature_cols)}")
for c in feature_cols:
    print(f"    {c}: {df_all[c].dtype}")

# ── sample rows ───────────────────────────────────────────────────────────────
print("\n[6] Sample 5 rows (one per store, mid-series day):")
sample_rows = []
for store in ["CA_1", "TX_1", "WI_1", "CA_3", "TX_3"]:
    row = df_all[(df_all["store_id"] == store) & (df_all["d_num"] == 500)].iloc[0]
    sample_rows.append(row)
sample_df = pd.DataFrame(sample_rows)[["id", "d_num", "sales", "lag_7", "lag_28",
                                        "roll_mean_28", "roll_std_28",
                                        "is_weekend", "is_holiday", "is_snap_ca",
                                        "sell_price", "price_change_pct"]]
print(sample_df.to_string(index=False))

print("\n" + "=" * 60)
print("DAY 5 COMPLETE")
print(f"  Rows: {len(df_all):,}")
print(f"  Columns: {len(df_all.columns)} total ({len(feature_cols)} features)")
print(f"  Output: {FEAT_DIR}")
print(f"  Total size: {total_mb:.0f} MB")
print("=" * 60)
