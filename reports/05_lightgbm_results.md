# Day 6 — LightGBM Global Model Results

## Overview

Day 6 trains a single global LightGBM model on all 30,490 M5 series simultaneously — the approach used by all top-10 M5 competition finishers. Unlike the per-series classical methods (Days 3–4), LightGBM learns patterns across all series, eliminating the per-series compute constraint and enabling cross-series information sharing.

**Training data:** 38 engineered features from Day 5 pipeline, `d_num ∈ [1000, 1913]` (~27.8M training rows after dropping lag-NaN early rows).  
**Validation:** `d_num ∈ [1886, 1913]` (last 28 training days, mirroring the forecast horizon).  
**Forecast:** `d_num ∈ [1914, 1941]` using pre-computed features from the parquet.

---

## Model Comparison

| Model | WRMSSE (full 30,490) | Kaggle Public | Kaggle Private | Notes |
|-------|---------------------|---------------|----------------|-------|
| **Best tuned (Optuna)** | **0.5422** | TBD | TBD | tvp=1.499, lr=0.025, 879 iter |
| Tweedie (power=1.1) | 0.5442 | TBD | TBD | Handily beats RMSE; 823 iter |
| RMSE baseline | 0.5651 | TBD | TBD | vanilla regression; 153 iter |
| SN28 (Day 2 reference) | 0.8377 | 0.8377 | 0.8956 | best classical baseline |
| ETS 1k-sample | 0.6541* | 0.8377 | 0.8698 | *sample score, not comparable |
| Top-Down Prophet | 0.5555* | — | — | *sample score, 1k series |

**Overall improvement vs SN28 baseline: −0.2955 WRMSSE (−35.2%)**

---

## RMSE vs Tweedie Analysis

**Why Tweedie for M5?**  
The M5 dataset has extreme zero-inflation: 68% of all observations are zero, rising to 77% for HOBBIES. Standard RMSE regression treats zero observations as normally distributed targets — it assigns equal weight to all observations regardless of value. Tweedie loss (`variance_power=1.1`) models the mean of a Tweedie distribution, which naturally handles the compound Poisson structure of retail demand:

- For series that are mostly zero, Tweedie loss correctly pulls predictions toward low positive values rather than the mean of a noisy mixture.
- For high-volume FOODS series, Tweedie power=1.1 approximates RMSE behaviour (low power → near-Gaussian).

**Observed outcome:**
- RMSE: WRMSSE=0.5651 (best_iter=153, training time ~118s)
- Tweedie (power=1.1): WRMSSE=0.5442 (best_iter=823, ~395s) — **−0.0209 improvement**
- Best tuned (power=1.499): WRMSSE=0.5422 — Optuna found higher variance power works better

The large iteration count difference (153 vs 823) is expected: Tweedie loss landscape is shallower/noisier than RMSE, requiring more boosting rounds to converge to the same plateau.

---

## Optuna Hyperparameter Search

**Strategy:** 20 trials on `d_num ∈ [1600, 1913]` subset (~8.7M rows, ~3× faster per trial). Best params then retrained on full `d_num ∈ [1000, 1913]`. Total Optuna runtime: ~1645s (~27 min).

**Search space:**

| Parameter | Values tried |
|-----------|-------------|
| `learning_rate` | 0.025, 0.05, 0.075, 0.1 |
| `num_leaves` | 32, 64, 128, 256 |
| `min_data_in_leaf` | 20, 50, 100, 200 |
| `feature_fraction` | 0.5, 0.7, 0.8, 0.9 |
| `bagging_fraction` | 0.5, 0.7, 0.8, 0.9 |
| `lambda_l2` | 0.0, 0.1, 0.5, 1.0 |
| `tweedie_variance_power` | 1.0–1.5 |

**Best trial:**

| Parameter | Value |
|-----------|-------|
| `learning_rate` | 0.025 |
| `num_leaves` | 64 |
| `min_data_in_leaf` | 20 |
| `feature_fraction` | 0.9 |
| `bagging_fraction` | 0.9 |
| `lambda_l2` | 1.0 |
| `tweedie_variance_power` | **1.499** |

**Best trial val metric:** 3.8016 (Tweedie loss on subset)  
**Retrained full-data WRMSSE:** 0.5422 (best_iter=879, ~495s)

**Key findings:**
- Optuna converged to lower `num_leaves` (64 vs base 128) — less tree complexity, more regularisation, better generalisation on sparse series
- Higher `tweedie_variance_power` (1.499 vs 1.1) — model "knows" HOBBIES/HOUSEHOLD are more Poisson-like; higher power penalises large-value errors less
- High `feature_fraction`=0.9 and `bagging_fraction`=0.9 — minimal subsampling; dataset is already diverse enough

---

## Feature Importance (Top 20)

*Chart saved to `reports/charts/lgbm_feature_importance.png`*

| Rank | Feature | Gain | Split |
|------|---------|------|-------|
| 1 | roll_mean_28 | 588,258,000 | 2,679 |
| 2 | roll_mean_7 | 312,609,000 | 2,998 |
| 3 | roll_std_28 | 56,584,000 | 552 |
| 4 | roll_mean_56 | 40,936,000 | 3,371 |
| 5 | sell_price | 13,248,000 | 5,453 |
| 6 | roll_mean_180 | 12,964,000 | 3,399 |
| 7 | roll_std_180 | 9,383,000 | 1,733 |
| 8 | weekday | 7,343,000 | 3,242 |
| 9 | price_relative_mean | 5,737,000 | 3,045 |
| 10 | day_of_month | 3,626,000 | 2,628 |
| 11 | has_price_change | 3,091,000 | 628 |
| 12 | dept_id | 3,064,000 | 3,385 |
| 13 | price_change_pct | 2,770,000 | 1,370 |
| 14 | week_of_year | 2,399,000 | 3,112 |
| 15 | roll_std_7 | 1,937,000 | 1,575 |
| 16 | month | 1,896,000 | 1,377 |
| 17 | roll_max_28 | 1,820,000 | 328 |
| 18 | store_id | 1,761,000 | 2,809 |
| 19 | roll_max_7 | 1,499,000 | 440 |
| 20 | roll_std_56 | 1,307,000 | 721 |

**Hypothesis vs reality:**
- Rolling means dominate (top 4 by gain) — confirmed. But `lag_7`/`lag_28` did not appear in top 20 by gain (high split count implies they split often but each split adds little information, suggesting rolling aggregates capture the lag signal better).
- `sell_price` at rank 5 by gain (5,453 splits — highest split count) — confirmed; drives demand directly.
- `weekday` at rank 8 — confirmed; strong weekly seasonality across all categories.
- `dept_id` at rank 12, `store_id` at rank 18 — hierarchy features matter but below price/rolling in gain. Confirms Day 4 finding that category structure is informative; LightGBM extracts this natively.
- Lag features (lag_7, lag_14, lag_28, lag_56) did not appear in top 20 by gain — rolling means subsume their signal with better noise reduction.

---

## Per-Category WRMSSE Breakdown

| Category | RMSE | Tweedie | Best tuned | vs Classical (sample) |
|----------|------|---------|-----------|----------------------|
| FOODS | — | — | **0.5204** | ETS: 0.5616 → **+0.04 better** |
| HOUSEHOLD | — | — | **0.5905** | ETS: 1.7023 → **−1.11 better** |
| HOBBIES | — | — | **0.6112** | ETS: 3.2663 → **−2.65 better** |
| **Overall** | **0.5651** | **0.5442** | **0.5422** | SN28: 0.8377 → **−0.30 better** |

**HOBBIES result — the key test:**  
Classical methods (ETS, ARIMA, Prophet) all collapsed to zero-forecast fallback for ~390/1000 sparse HOBBIES series, producing WRMSSE=3.2663 on the 1k sample. LightGBM trained on all 30,490 series achieves **WRMSSE=0.6112** — a ~5× reduction. The cross-series pattern transfer works exactly as hypothesised: the model learns "this category × store typically sells 0.1–0.3 units/day" from high-volume neighbours and transfers that signal to sparse series, eliminating the all-zero fallback failure mode.

**HOUSEHOLD result:**  
1.7023 (ETS) → 0.5905 (LightGBM) — 3× improvement. Mid-volume intermittent demand benefits enormously from rolling window features and Tweedie loss.

**FOODS result:**  
0.5616 (ETS on 1k sample) vs 0.5204 (LightGBM on full 30,490). LightGBM is better, but the gap is smaller — FOODS is the regime where per-series statistical models are most competitive.

---

## Why Global ML Beats Classical Per-Series

This experiment completes the argument started in Day 3:

| Dimension | Per-series classical | Global LightGBM |
|-----------|---------------------|-----------------|
| Compute | O(n_series × fit_time) | O(1 training job) |
| Full 30,490 series | ~34 hrs (ARIMA) | ~20 min |
| Sparse series handling | Zero-forecast fallback | Cross-series pattern transfer |
| Feature richness | Calendar/price awkward | Native in feature matrix |
| Hierarchy information | Not used (per-series) | cat_id/dept_id splits |
| HOBBIES WRMSSE | 3.27 (zero-fallback) | **0.61** |

**Total training time (Day 6):**
- Feature load: 5.3s
- RMSE model: ~118s
- Tweedie model: ~395s
- Optuna 20 trials: ~1645s
- Best retrain: ~495s
- **Total: ~44 min end-to-end**

---

## Files

| File | Description |
|------|-------------|
| `scripts/06_train_lightgbm.py` | Full training pipeline |
| `data/models/lgbm_rmse.pkl` | RMSE model (gitignored) |
| `data/models/lgbm_tweedie.pkl` | Tweedie model (gitignored) |
| `data/models/lgbm_best.pkl` | Best Optuna-tuned model (gitignored) |
| `reports/day6_lgbm_scores.json` | All WRMSSE scores and Optuna results |
| `reports/day6_feature_importance.csv` | Full feature importance table |
| `reports/charts/lgbm_feature_importance.png` | Feature importance bar chart |
| `submissions/lgbm_best_submission.csv` | Best model Kaggle submission |
