# Sample Selection for Day 3 Classical Methods Evaluation

## Overview

The M5 dataset contains 30,490 base series (store × item level). Running ARIMA/SARIMA on all 30,490 series would take approximately 24–36 hours locally (SARIMA fits average ~3s/series × 30,490 = ~25 hours). A stratified 1,000-series sample was selected to make the evaluation tractable in under 2 hours while preserving representativeness across demand regimes.

## Stratification Design

Rather than sampling uniformly at random (which would be dominated by FOODS, the largest category), the sample is stratified by category with deliberate **volume targeting per stratum**:

| Category | N sampled | Selection rule | Rationale |
|----------|-----------|----------------|-----------|
| FOODS | 334 | Top 334 by total training sales (descending) | High-volume items dominate revenue weights; critical for WRMSSE accuracy |
| HOUSEHOLD | 333 | 333 series closest to median sales rank within HOUSEHOLD | Middle-volume items; representative of the modal forecasting difficulty |
| HOBBIES | 333 | Bottom 333 by total training sales (ascending) | Low-volume / sparse items; tests model robustness on intermittent demand |

**Total training sales** is computed as the sum of daily sales over d_1 through d_1913 (the full training window used for forecasting d_1914–d_1941).

## Why This Stratification Is Representative

1. **Demand regime coverage**: The M5 dataset has a known bimodal demand structure — high-volume staples (FOODS) and intermittent low-volume items (HOBBIES). By deliberately over-sampling the extremes relative to a pure random draw, the sample forces each model to be evaluated across the full difficulty spectrum.

2. **WRMSSE alignment**: Because WRMSSE uses dollar-revenue weights, high-volume FOODS items receive the largest weight in any full-dataset evaluation. Capturing the top FOODS sellers ensures the sample's WRMSSE is driven by economically significant series, not noise from rarely-sold items.

3. **Sparse-series stress test**: The bottom-333 HOBBIES selection includes the most zero-heavy series in the dataset. This is where statistical models (especially SARIMA) most commonly fall back to naive forecasts — making it the most informative stratum for diagnosing model failures.

4. **Middle-ground calibration**: The HOUSEHOLD median-rank selection avoids cherry-picking and ensures the sample contains series where all methods compete on roughly equal footing.

## Important Caveat: Sample WRMSSE vs Full-Dataset WRMSSE

Scores computed on this 1,000-series subset are **not directly comparable** to full-dataset WRMSSE values from Day 2. Differences arise because:

- The hierarchical aggregation is computed only over the 1,000 sampled series (not all 30,490)
- Revenue weights are re-normalized within the subset
- Aggregate levels (level 1–9) reflect only the sampled series' totals

To enable relative comparison, the Day 3 leaderboard includes the **Seasonal Naive 28 WRMSSE computed on the same 1,000-series subset** as a reference point. This makes the "delta vs best baseline" meaningful even though the absolute scores differ from Day 2.

## Files

- `reports/sample_1000_series.csv` — the 1,000 selected series IDs with metadata and stratum labels
- `data/processed/sample_1000_series.csv` — same file, also written to data/processed (gitignored, present locally)
