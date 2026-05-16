"""
Post-processing: reads day3_classical_scores.json and writes:
  - reports/leaderboard.md (updated)
  - reports/02_classical_methods.md
"""
from __future__ import annotations

import sys
import os
import json

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(PROJ_ROOT, "reports")

results_path = os.path.join(REPORTS_DIR, "day3_classical_scores.json")
if not os.path.exists(results_path):
    print(f"ERROR: {results_path} not found. Run run_day3.py first.")
    sys.exit(1)

with open(results_path) as f:
    data = json.load(f)

sn28 = data["seasonal_naive_28_reference"]
methods = data["methods"]

sn28_wrmsse = sn28["wrmsse_sample"]
sn28_cat = sn28["wrmsse_by_category"]

print(f"Seasonal Naive 28 (1k sample): {sn28_wrmsse:.4f}")
for m, r in methods.items():
    print(f"{m.upper()}: {r['wrmsse_sample']:.4f}  fallbacks={r['n_fallbacks']}  "
          f"time={r['wall_time_seconds']:.1f}s")

# ── leaderboard ───────────────────────────────────────────────────────────────

METHOD_DISPLAY = {
    "ets": "ETS (Holt-Winters add/add, m=7)",
    "arima": "ARIMA (auto, non-seasonal)",
    "sarima": "SARIMA (auto, m=7, D=1)",
    "sarimax": "SARIMAX (m=7 + calendar/price exog)",
}
METHOD_FAMILY = {m: "Classical Statistical" for m in METHOD_DISPLAY}

# Sort by wrmsse ascending for ranking
all_method_results = []
for m, r in methods.items():
    all_method_results.append((m, r["wrmsse_sample"]))
all_method_results.sort(key=lambda x: x[1])

# Build the new leaderboard entries (formatted rows for the full table)
# Ranks 1-7 are existing baselines; classical methods start at rank 8 potentially
# We'll re-rank everything together

existing_rows = [
    # (wrmsse_local, kaggle_lb, name, family, day, notes)
    (0.8835, "**0.8377**", "Seasonal Naive (28-day)", "Baseline", 2, "Best baseline; repeats last 28 days"),
    (0.9128, "—", "Seasonal Naive (7-day)", "Baseline", 2, "Repeats last week's pattern"),
    (1.1183, "—", "Moving Average (28d)", "Baseline", 2, "Mean of last 28 days"),
    (1.1374, "—", "Moving Average (90d)", "Baseline", 2, "Mean of last 90 days"),
    (1.1721, "—", "Moving Average (7d)", "Baseline", 2, "Mean of last 7 days"),
    (1.5111, "—", "Seasonal Naive (365-day)", "Baseline", 2, "Same window last year"),
    (1.5137, "—", "Naive (last value)", "Baseline", 2, "Repeat last observation"),
]

leaderboard_content = f"""# ShelfSense M5 — Model Leaderboard

Metric: **WRMSSE** (lower is better).
Validation period: **d_1914 – d_1941** (last 28 days of `sales_train_evaluation.csv`).
Kaggle public LB: validation period submitted to M5 Forecasting Accuracy competition.

---

## Full-Dataset Models (30,490 series)

| Rank | Model | Family | WRMSSE (local) | Kaggle LB | Day | Notes |
|------|-------|--------|---------------|-----------|-----|-------|
| 1 | Seasonal Naive (28-day) | Baseline | **0.8835** | **0.8377** | 2 | Best baseline; repeats last 28 days |
| 2 | Seasonal Naive (7-day) | Baseline | 0.9128 | — | 2 | Repeats last week's pattern |
| 3 | Moving Average (28d) | Baseline | 1.1183 | — | 2 | Mean of last 28 days |
| 4 | Moving Average (90d) | Baseline | 1.1374 | — | 2 | Mean of last 90 days |
| 5 | Moving Average (7d) | Baseline | 1.1721 | — | 2 | Mean of last 7 days |
| 6 | Seasonal Naive (365-day) | Baseline | 1.5111 | — | 2 | Same window last year |
| 7 | Naive (last value) | Baseline | 1.5137 | — | 2 | Repeat last observation |

---

## Day 3: Classical Statistical Methods — 1,000-Series Sample

> **Note**: Scores below are computed on a stratified 1,000-series sample (not all 30,490).
> They are **not directly comparable** to the full-dataset WRMSSE above.
> Use the "SN28 on same 1k" column to gauge relative improvement vs the best baseline.
> Seasonal Naive 28 on the same 1k sample: **{sn28_wrmsse:.4f}**

| Rank | Model | Family | WRMSSE sample (1k) | SN28 on same 1k | Kaggle LB | Day | Notes |
|------|-------|--------|-------------------|-----------------|-----------|-----|-------|
"""

# Add classical methods ranked by sample wrmsse
rank = 1
for m, wrmsse in all_method_results:
    r = methods[m]
    delta = wrmsse - sn28_wrmsse
    delta_str = f"{delta:+.4f} vs SN28"
    cat = r.get("wrmsse_by_category", {})
    fb = r["n_fallbacks"]
    t_per = r.get("wall_time_per_series", r["wall_time_seconds"] / 1000)
    notes = f"{delta_str}; {fb} fallbacks; {t_per:.1f}s/series"
    leaderboard_content += (
        f"| {rank} | {METHOD_DISPLAY[m]} | Classical Statistical "
        f"| **{wrmsse:.4f}** | {sn28_wrmsse:.4f} | — | 3 | {notes} |\n"
    )
    rank += 1

# Category-level breakdown table for each method
leaderboard_content += f"""
---

## Day 3: Category-Level WRMSSE (1k sample)

| Model | FOODS (top vol.) | HOUSEHOLD (mid vol.) | HOBBIES (low vol.) |
|-------|-----------------|---------------------|-------------------|
| Seasonal Naive 28 (ref) | {sn28_cat.get('FOODS', 0):.4f} | {sn28_cat.get('HOUSEHOLD', 0):.4f} | {sn28_cat.get('HOBBIES', 0):.4f} |
"""

for m, wrmsse in all_method_results:
    r = methods[m]
    cat = r.get("wrmsse_by_category", {})
    leaderboard_content += (
        f"| {METHOD_DISPLAY[m].split('(')[0].strip()} | "
        f"{cat.get('FOODS', 0):.4f} | "
        f"{cat.get('HOUSEHOLD', 0):.4f} | "
        f"{cat.get('HOBBIES', 0):.4f} |\n"
    )

# Seasonal Naive level breakdown (from Day 2)
leaderboard_content += """
---

## Seasonal Naive (28-day) — Level Breakdown (full dataset, Day 2)

| Level | Groups | WRMSSE |
|-------|--------|--------|
| level_1 (total) | 1 | 0.6289 |
| level_2 (state) | 3 | 0.6892 |
| level_3 (store) | 10 | 0.7486 |
| level_4 (category) | 3 | 0.6565 |
| level_5 (department) | 7 | 0.7573 |
| level_6 (state × cat) | 9 | 0.7191 |
| level_7 (state × dept) | 21 | 0.8042 |
| level_8 (store × cat) | 30 | 0.7938 |
| level_9 (store × dept) | 70 | 0.8758 |
| level_10 (item) | 3 049 | 1.2525 |
| level_11 (state × item) | 9 147 | 1.3027 |
| level_12 (store × item) | 30 490 | 1.3731 |
| **Total WRMSSE** | — | **0.8835** |

---

## How to Read This Table

- **WRMSSE (local)**: computed with `src/evaluation/wrmsse.py` on d_1914–d_1941 actuals
- **Kaggle LB**: public leaderboard score after CSV submission to the competition
- **Sample WRMSSE (1k)**: computed on a stratified 1,000-series subset — not directly comparable to full-dataset WRMSSE
- **Calibration note**: local full-dataset scores are ~5% pessimistic vs Kaggle (observed: local 0.8835 vs Kaggle 0.8377 for Seasonal Naive 28)

## Scoring Notes

- WRMSSE = unweighted mean of 12 hierarchical level scores
- Weights = dollar revenue in last 28 training days, normalised per level
- Scale = naive-1 MSE on full training history, per aggregated series
- Aggregate levels (1–9) always score lower than item levels (10–12) due to noise cancellation
"""

leaderboard_path = os.path.join(REPORTS_DIR, "leaderboard.md")
with open(leaderboard_path, "w") as f:
    f.write(leaderboard_content)
print(f"Wrote {leaderboard_path}")

# ── methods report ────────────────────────────────────────────────────────────

# Determine category winner per method
def best_cat(cat_scores):
    if not cat_scores:
        return "N/A"
    return min(cat_scores, key=cat_scores.get)

# Find which method wins each category
cat_winners = {}
for cat in ["FOODS", "HOUSEHOLD", "HOBBIES"]:
    best_m = None
    best_s = float("inf")
    for m, r in methods.items():
        s = r.get("wrmsse_by_category", {}).get(cat, float("inf"))
        if s < best_s:
            best_s = s
            best_m = m
    cat_winners[cat] = (best_m, best_s)

# Build timing table
timing_rows = []
for m, r in methods.items():
    w = r["wall_time_seconds"]
    per_s = r.get("wall_time_per_series", w / 1000)
    timing_rows.append((m.upper(), w, per_s, r["n_fallbacks"], r["n_zero_forecasts"]))

best_method = all_method_results[0][0]
best_wrmsse = all_method_results[0][1]

# ETS vs ARIMA analysis
ets_score = methods["ets"]["wrmsse_sample"]
arima_score = methods["arima"]["wrmsse_sample"]
sarima_score = methods["sarima"]["wrmsse_sample"]
sarimax_score = methods["sarimax"]["wrmsse_sample"]

ets_win = "ETS" if ets_score < arima_score else "ARIMA"
sarimax_win = "SARIMAX" if sarimax_score < sarima_score else "SARIMA"

methods_report = f"""# Day 3: Classical Statistical Forecasting Methods

## Summary

Four classical statistical methods were implemented and evaluated on a stratified 1,000-series sample from the M5 dataset.

**Key finding**: The best classical method ({METHOD_DISPLAY[best_method]}) achieved sample WRMSSE of **{best_wrmsse:.4f}** vs Seasonal Naive 28's **{sn28_wrmsse:.4f}** on the same sample (delta: {best_wrmsse - sn28_wrmsse:+.4f}).

---

## Why a 1,000-Series Sample?

Running ARIMA on all 30,490 M5 series is locally infeasible:
- ARIMA average fit time: ~{methods['arima'].get('wall_time_per_series', 0):.1f}s/series
- Projected full-dataset time: 30,490 × {methods['arima'].get('wall_time_per_series', 0):.1f}s ≈ {30490 * methods['arima'].get('wall_time_per_series', 0) / 3600:.1f} hours
- SARIMA on full dataset: 30,490 × {methods['sarima'].get('wall_time_per_series', 0):.1f}s ≈ {30490 * methods['sarima'].get('wall_time_per_series', 0) / 3600:.1f} hours

A stratified 1,000-series sample reduces runtime to under 2 hours while covering all demand regimes. See `reports/02_sample_selection.md` for the full sampling rationale.

---

## Timing Results

| Method | Total wall time (s) | Per-series (s) | Fallbacks | Zero forecasts |
|--------|--------------------|--------------|-----------|----|
"""
for m_name, w, per_s, fb, zf in timing_rows:
    methods_report += f"| {m_name} | {w:.1f} | {per_s:.2f} | {fb} | {zf} |\n"

methods_report += f"""
---

## WRMSSE Results (1,000-series sample)

> Reference: Seasonal Naive 28 on same 1k sample = **{sn28_wrmsse:.4f}**

| Method | Sample WRMSSE | Delta vs SN28 |
|--------|--------------|---------------|
"""
for m, wrmsse in all_method_results:
    delta = wrmsse - sn28_wrmsse
    sign = "+" if delta >= 0 else ""
    methods_report += f"| {METHOD_DISPLAY[m]} | {wrmsse:.4f} | {sign}{delta:.4f} |\n"

methods_report += f"""
---

## Category-Level Results

| Method | FOODS (top vol.) | HOUSEHOLD (mid) | HOBBIES (low vol.) |
|--------|-----------------|----------------|-------------------|
| Seasonal Naive 28 (ref) | {sn28_cat.get('FOODS', 0):.4f} | {sn28_cat.get('HOUSEHOLD', 0):.4f} | {sn28_cat.get('HOBBIES', 0):.4f} |
"""
for m, _ in all_method_results:
    r = methods[m]
    cat = r.get("wrmsse_by_category", {})
    methods_report += (
        f"| {METHOD_DISPLAY[m].split('(')[0].strip()} | "
        f"{cat.get('FOODS', 0):.4f} | "
        f"{cat.get('HOUSEHOLD', 0):.4f} | "
        f"{cat.get('HOBBIES', 0):.4f} |\n"
    )

# Identify winners
methods_report += f"""
### Category Winners

- **FOODS** (high-volume): {cat_winners['FOODS'][0].upper() if cat_winners.get('FOODS') else 'N/A'} wins with {cat_winners.get('FOODS', (None, 0))[1]:.4f} (SN28 ref: {sn28_cat.get('FOODS', 0):.4f})
- **HOUSEHOLD** (mid-volume): {cat_winners['HOUSEHOLD'][0].upper() if cat_winners.get('HOUSEHOLD') else 'N/A'} wins with {cat_winners.get('HOUSEHOLD', (None, 0))[1]:.4f} (SN28 ref: {sn28_cat.get('HOUSEHOLD', 0):.4f})
- **HOBBIES** (low-volume): {cat_winners['HOBBIES'][0].upper() if cat_winners.get('HOBBIES') else 'N/A'} wins with {cat_winners.get('HOBBIES', (None, 0))[1]:.4f} (SN28 ref: {sn28_cat.get('HOBBIES', 0):.4f})

---

## When Does ETS Beat ARIMA?

ETS ({ets_score:.4f}) {"outperforms" if ets_score < arima_score else "underperforms"} ARIMA ({arima_score:.4f}) overall (delta: {ets_score - arima_score:+.4f}).

**ETS tends to beat ARIMA when:**
- The series has a clear additive trend + weekly seasonality with stable variance — ETS's explicit trend/seasonal decomposition fits without needing to search an ARIMA order space
- The series is short or dense enough that the exponential weighting adapts quickly to level shifts
- ARIMA's auto-order selection converges to a poor local optimum due to non-stationarity after differencing

**ARIMA tends to beat ETS when:**
- The series has irregular seasonality or multiple seasonal drivers that ETS's fixed m=7 cannot capture
- The series is nearly stationary (no trend), where ARIMA(0,0,q) or AR(p) structures fit naturally
- ETS's initialization estimate is poor for very sparse or highly irregular series

---

## When Does SARIMAX Beat SARIMA?

SARIMAX ({sarimax_score:.4f}) {"outperforms" if sarimax_score < sarima_score else "does not outperform"} SARIMA ({sarima_score:.4f}) overall (delta: {sarimax_score - sarima_score:+.4f}).

**SARIMAX exogenous features used:**
- `is_holiday`: 1 if a named event is in the M5 calendar on that day
- `snap`: state-specific SNAP benefit disbursement flag (CA/TX/WI)
- `is_weekend`: 1 if wday ∈ {{1, 7}} (Sunday/Saturday)
- `month`: integer 1–12 (seasonal baseline)
- `sell_price`: last-known price forward-filled from sell_prices.csv

**SARIMAX tends to beat SARIMA when:**
- Items show strong holiday effects (food items near Thanksgiving, Christmas)
- SNAP disbursement creates observable demand spikes for FOODS/HOUSEHOLD staples
- Price changes drive demand shifts that the seasonal component alone cannot explain

**SARIMA can outperform SARIMAX when:**
- The exogenous features are noisy or uncorrelated with demand for that specific item
- The exog estimation adds coefficient variance that exceeds the bias reduction from including the features
- Series are sparse — exog coefficients are poorly identified with few non-zero observations

---

## Fallback Analysis

Methods fall back to simpler forecasts when:
1. **>80% zeros**: Series with more than 80% zero training observations receive a zero forecast directly (no fitting attempted). This covers the most intermittent HOBBIES items.
2. **Fitting failure**: If auto_arima raises any exception, ARIMA falls back to the last-value naive; SARIMA/SARIMAX fall back to 7-day seasonal naive.
3. **ETS failure**: If HoltWinters with trend+seasonal fails (e.g., series too short or constant), falls back to simple exponential smoothing, then to last-value.

High fallback rates in SARIMA/SARIMAX are expected — constrained SARIMA (max_p=2, m=7, D=1) frequently fails to converge on very sparse or zero-dominated series that pass the 80% threshold but still have irregular structure.

---

## Limitations

1. **Sample representativeness**: The 1,000-series sample over-represents high-volume FOODS and low-volume HOBBIES by design. The sample WRMSSE may not match full-dataset performance for methods that behave differently on mid-range items.

2. **WRMSSE scoring caveat**: All Day 3 scores are computed on the 1,000-series subset with re-normalized weights. Aggregate levels (1–9) are aggregations of only these 1,000 series, not the full hierarchy. Scores are not comparable to Day 2 full-dataset WRMSSE.

3. **WRMSSE evaluator version**: The `src/evaluation/wrmsse.py` evaluator may receive bug fixes after Day 3 scoring. Scores in this report may be slightly adjusted when the evaluator is updated.

4. **Hyperparameter constraints**: SARIMA is constrained to max_p=2, max_P=1 for speed. The true optimal order may exceed these bounds for some series, potentially underestimating SARIMA's ceiling performance.

5. **Exogenous feature quality**: SARIMAX's `sell_price` is forward-filled at the weekly granularity from `sell_prices.csv`. This introduces stale-price periods when items are temporarily de-listed, which may degrade the exog signal for some series.

6. **No cross-validation**: Model selection uses a single train/validation split (d_1–d_1913 train, d_1914–d_1941 validate). CV-based selection would give more robust estimates at the cost of further increased runtime.
"""

methods_report_path = os.path.join(REPORTS_DIR, "02_classical_methods.md")
with open(methods_report_path, "w") as f:
    f.write(methods_report)
print(f"Wrote {methods_report_path}")
print("Done!")
