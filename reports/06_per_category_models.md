# Per-Category LightGBM + Recursive Evaluation Forecast

## Overview

Extends the global LightGBM in two ways:
1. **Recursive evaluation forecast** — generates predictions for the true evaluation
   period d_1942–d_1969 (Kaggle private LB), fixing the private score gap.
2. **Per-category models** — separate LightGBM + Optuna for FOODS, HOUSEHOLD, HOBBIES,
   testing whether category-specific Tweedie tuning improves over the global model.

---

## Sanity Check: Recursive vs Single-Step (Validation Period)

| Forecast method | WRMSSE (val d_1914–1941) |
|-----------------|------------------------|
| Global single-step | 0.5422 |
| Global recursive | 0.6019 |
| Gap | 0.0597 (11% relative) |

Single-step uses pre-computed features from actual sales — strictly better than recursive.
The gap quantifies error propagation over 28 recursive steps. For sparse M5 series with
high zero rates, a 6–12% gap over 28 steps is expected and validates the buffer logic is
working correctly (if recursive = single-step exactly, predictions would not be feeding back).

---

## Per-Category Optuna Results

| Category | tvp | lr | num_leaves | Val WRMSSE | vs Global (0.5422) |
|----------|-----|----|------------|-----------|-------------------|
| FOODS | 1.438 | 0.05 | 256 | 0.5129 | −0.0293 (better) |
| HOUSEHOLD | 1.555 | 0.025 | 256 | 0.7051 | +0.1629 (worse) |
| HOBBIES | 1.402 | 0.075 | 32 | 0.6502 | +0.1080 (worse) |

**Hypothesis vs reality:**

| Category | Expected tvp | Actual tvp | As expected? |
|----------|-------------|-----------|--------------|
| FOODS (dense, 62% zeros) | low (1.0–1.3) | 1.438 | No — higher than expected |
| HOUSEHOLD (moderate intermittency) | medium (1.2–1.6) | 1.555 | Yes |
| HOBBIES (sparse, 77% zeros) | high (1.5–1.9) | 1.402 | No — lower than HOUSEHOLD |

The hypothesis that HOBBIES would require the highest tvp did not hold. Likely explanation:
with only 5,650 HOBBIES series (vs 14,370 FOODS), the per-category dataset is too small
to reliably estimate tvp — Optuna's 10 trials have high variance on the smaller dataset.
The global model's cross-series data (all 30,490 series) produces a more reliable tvp estimate.

---

## Validation Period Comparison (d_1914–1941)

| Model | WRMSSE (full 30,490) | vs global |
|-------|---------------------|-----------------|
| Global LightGBM | **0.5422** | baseline |
| Per-category LightGBM | 0.5726 | +0.0304 (worse) |
| Blend (0.6×per_cat + 0.4×global) | 0.5545 | +0.0123 (worse) |

**Per-category models are worse on validation.** Splitting the dataset by category removes
cross-series information: a HOBBIES model can no longer observe that "a store with high FOODS
demand also tends to sell HOUSEHOLD items regularly." The global model's tree splits on
`cat_id`/`dept_id` already handle category differences internally without data fragmentation.

---

## Kaggle Scores (Private LB Surprise)

| Submission | Local val WRMSSE | Kaggle Public | Kaggle Private |
|------------|-----------------|---------------|----------------|
| lgbm_best (SN28-filled eval) | 0.5422 | 0.5422 | 0.8956 |
| lgbm_global_recursive | 0.5422 | 0.5422 | **0.8138** |
| **lgbm_blend** | 0.5545 | 0.5545 | **0.7126** |

**Key findings:**

1. **Recursive eval forecast works:** global private dropped 0.8956 → 0.8138 (−0.082). Direct
   payoff from generating proper d_1942–d_1969 predictions instead of using SN28 as placeholder.

2. **Blend beats global on private despite worse validation:** 0.7126 vs 0.8138 (−0.101 private
   improvement). Classic ensemble diversity effect: per-category and global models make
   different prediction errors. The evaluation period (d_1942–d_1969) is structurally different
   from validation (d_1914–d_1941) — ensemble diversity generalises better across the gap
   than either model alone.

3. **Public vs private divergence:** blend is worse on public (0.5545 vs 0.5422) but much
   better on private. This confirms that optimising for the validation window alone is
   insufficient — model diversity matters more for out-of-window generalisation.

---

## Why the Blend Wins on Private

The blend is 60% per-category + 40% global model predictions. Even though per-category models
score worse on the validation period, they capture a different manifold of the data:

- Per-category models have less data → higher bias, lower variance within-category patterns
- Global model has more data → lower bias, but may overfit to validation-period idiosyncrasies
- The 28-day evaluation period (d_1942–d_1969) represents genuinely new patterns
  (different calendar events, potential demand shifts) that the average of two different models
  predicts better than either alone

This is an interview-worthy finding: **"My per-category models scored worse on validation but
helped the ensemble. The lesson is that ensemble diversity is worth more than individual model
accuracy when the forecast horizon shifts."**

---

## Files

| File | Description |
|------|-------------|
| `scripts/07_train_per_category.py` | Full training + recursive forecast pipeline |
| `src/models/recursive_forecast.py` | Recursive forecast library |
| `data/models/lgbm_per_category_{FOODS,HOUSEHOLD,HOBBIES}.pkl` | Trained models (gitignored) |
| `reports/day7_scores.json` | All local scores and parameters |
| `submissions/lgbm_global_recursive.csv` | Global model, full eval period (public 0.5422, private 0.8138) |
| `submissions/lgbm_per_cat.csv` | Per-category models |
| `submissions/lgbm_blend.csv` | Best submission: blend (public 0.5545, private 0.7126) |
