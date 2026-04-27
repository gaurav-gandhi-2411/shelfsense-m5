# Day 4 — Hierarchical Forecasting Analysis

## Overview

This document reports the bottom-up vs top-down comparison using Prophet (cps=0.1) on the 1,000-series sample. The central finding is a large and consistent win for top-down approaches: forecasting at aggregate levels and then disaggregating to base series reduces WRMSSE by 0.10–0.11 versus fitting independent per-series models.

---

## Methods

### Bottom-Up (BU)

Fit Prophet independently on each of the 1,000 base series (item × store). Forecasts are aggregated by summing — but since WRMSSE is computed at each level separately, the BU score reflects the quality of base-series predictions propagated upward.

**Hyperparameters:** cps=0.1 (best from Day 4 sweep), multiplicative weekly seasonality, M5 holidays.

### Top-Down (TD)

For each aggregate grouping, sum the base-series training values to form a single aggregate series, fit one Prophet model, then disaggregate the aggregate forecast to base series using **last-28-day proportion shares**:

```
share[i] = sales_last28[i] / sum(sales_last28[group])
base_forecast[i] = aggregate_forecast * share[i]
```

Four aggregation levels were tested:

| Level | Groups in sample | Description |
|-------|-----------------|-------------|
| national | 1 | Sum of all 1,000 series |
| state | 3 | CA, TX, WI |
| category | 3 | FOODS, HOUSEHOLD, HOBBIES |
| dept | 7 | FOODS_1/2/3, HOUSEHOLD_1/2, HOBBIES_1/2 |

---

## Results

| Approach | WRMSSE (1k sample) | vs BU |
|----------|-------------------|-------|
| SN28 reference | 0.6778 | — |
| **Top-Down (category)** | **0.5555** | **−0.1083** |
| Top-Down (dept) | 0.5565 | −0.1073 |
| Top-Down (national) | 0.5580 | −0.1058 |
| Top-Down (state) | 0.5740 | −0.0898 |
| Bottom-Up Prophet | 0.6638 | 0 (ref) |
| ETS | 0.6541 | — |

**Every top-down variant beats every bottom-up model and every baseline we have.** The best result so far — TD category at 0.5555 — is 0.1223 better than the SN28 reference and 0.0986 better than ETS.

---

## Why Top-Down Wins So Decisively

### 1. Noise cancellation at the aggregate level

Base-level M5 series are extremely noisy — 68% zero rate, many intermittent series. When you sum 334 FOODS series into a single FOODS aggregate, the stochastic demand noise cancels. The aggregate shows a clean weekly seasonal pattern and a clear trend. Prophet fits this cleanly in a single model rather than trying (and often failing) to model 334 independent noise-dominated series.

### 2. Sparse-series problem disappears at the aggregate

390/1000 base series trigger the zero-forecast fallback in bottom-up models (>80% zeros). At the category level, no aggregate series is ever sparse — FOODS total, HOUSEHOLD total, and HOBBIES total all have continuous positive demand. Top-down eliminates the entire sparse-series failure mode.

### 3. Proportion shares are stable

The disaggregation assumption — that each series' share of its group is stable over the 28-day horizon — is approximately correct for M5. Items don't dramatically change their relative share of category sales over 4 weeks. The proportion-based disaggregation is a well-calibrated prior that is hard to beat without rich item-level features.

### 4. Category beats dept beats national — why?

This ordering is surprising at first (finer grouping wins over coarser), but makes sense:

- **Category > dept:** Category-level aggregates (FOODS, HOUSEHOLD, HOBBIES) are the most homogeneous groupings in M5 — series within a category share demand characteristics. Department aggregates are almost as good. National total is the noisiest aggregate (mixes all category patterns).
- **State is worst TD:** State aggregates mix categories (CA has proportionally more FOODS than WI), making the Prophet trend/seasonal fit less clean than pure-category aggregates.

The optimal grouping aligns with M5's natural demand structure: within-category demand is driven by the same consumer behaviour drivers.

---

## Implications for Global ML Models (Days 6–7)

The top-down result is the clearest signal in the project so far:

1. **Hierarchy-aware features matter.** Category, department, and state membership should be explicit features in LightGBM models — not just as embeddings but as the key dimensions that explain most of the variance.
2. **Grouped models may beat per-series models.** A LightGBM model trained per category (one model for FOODS, one for HOUSEHOLD, one for HOBBIES) might outperform a single global model — the category boundary is the strongest natural grouping.
3. **Proportion shares are a strong disaggregation baseline.** When reconciling hierarchical forecasts (Winsorized sum-product), using last-28-day shares as the disaggregation prior is simple and effective.

The practical priority for Day 6 LightGBM: include `cat_id` and `dept_id` as categorical features, and consider training per-category if the global model doesn't capture the category-level patterns well.

---

## Files

| File | Description |
|------|-------------|
| `src/models/prophet_model.py` | Prophet fit + batch runner + M5 holiday builder |
| `scripts/run_day4.py` | Changepoint sweep pipeline |
| `notebooks/04_prophet_hierarchical.ipynb` | Interactive analysis (charts, BU vs TD comparison) |
| `reports/day4_prophet_scores.json` | Per-cps scores and metadata from sweep |
| `reports/day4_hierarchical_scores.json` | BU vs TD level-by-level scores |
| `submissions/prophet_sample_submission.csv` | Prophet (cps=0.1) 1k-sample forecasts |
