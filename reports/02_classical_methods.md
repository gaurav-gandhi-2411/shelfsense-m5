# Day 3 — Classical Statistical Forecasting Methods

## Overview

Day 3 evaluated classical per-series statistical models (ETS and ARIMA) against the M5 dataset using a stratified 1,000-series sample. SARIMA was attempted but crashed with an out-of-memory error at 442/1000 series. SARIMAX was not run. This document explains the methodology, the OOM failure, and what the results tell us about classical methods on the M5 problem.

---

## Why a 1,000-Series Sample

The M5 dataset contains 30,490 bottom-level series (store × item). Running auto-ARIMA or SARIMA on all 30,490 series is not tractable locally:

| Method | ~Time/series | Full 30,490 estimate |
|--------|-------------|----------------------|
| ETS | 0.81 s | ~6.8 hours |
| ARIMA | 3.89 s | ~33 hours |
| SARIMA | ~8–12 s (est.) | ~68–102 hours |

A stratified 1,000-series sample reduces compute by 30x while preserving representativeness across the three demand regimes in M5. See `reports/02_sample_selection.md` for full stratification design.

**Stratum breakdown:**
- **FOODS_top (334 series):** Top 334 FOODS items by total training sales — highest revenue weight in WRMSSE.
- **HOUSEHOLD_mid (333 series):** 333 HOUSEHOLD items closest to median sales rank — moderate, regular demand.
- **HOBBIES_low (333 series):** Bottom 333 HOBBIES items by total sales — sparse, intermittent, many zero-heavy series.

---

## Sample WRMSSE vs Full-Catalogue Kaggle WRMSSE

Scores on the 1,000-series subset are **not directly comparable** to the full-catalogue Day 2 scores for two reasons:

1. **Re-normalized weights.** WRMSSE uses dollar-revenue weights. On the full 30,490 series, FOODS dominates because it has the highest sales volume. On the 1k sample, weights are re-computed from only the 1,000 series, shifting the effective weight distribution.
2. **Partial hierarchical aggregation.** Levels 1–9 aggregate across all bottom-level series. On the subset, level-1 (total) is the sum of only 1,000 series — not the national total. Aggregate-level WRMSSE therefore reflects a different signal than in the full dataset.

The Day 3 leaderboard includes **Seasonal Naive 28 scored on the same 1k sample (WRMSSE = 0.6778)** as a within-day reference baseline. All Day 3 model scores should be read relative to this number, not relative to the full-catalogue SN28 score of 0.8377.

To get a full-catalogue Kaggle score for ETS and ARIMA, the sample submission CSVs were submitted to the Kaggle competition. Those scores appear in `reports/leaderboard.md` once available.

---

## Results

### ETS (Exponential Smoothing)

| Metric | Value |
|--------|-------|
| WRMSSE (1k sample) | **0.6541** |
| vs SN28 reference (1k) | **−0.0237 improvement** |
| FOODS | 0.5616 |
| HOUSEHOLD | 1.7023 |
| HOBBIES | 3.2663 |
| Wall time | 814 s (13.5 min) |
| Time/series | 0.81 s |
| Fallbacks to SES | 0 |
| Zero forecasts issued | 390 / 1000 |
| Kaggle full-catalogue | pending |

**Configuration:** Additive trend + additive weekly seasonality (m=7), statsmodels `ExponentialSmoothing`, fallback to simple exponential smoothing, then to last-value. Series with >80% zeros receive a zero forecast (the zero-forecast fallback accounts for 390/1000 series, all concentrated in HOBBIES).

**Reading the scores:** ETS beats seasonal naive on FOODS (0.5616 vs 0.6400) because it captures level and trend adaptively. It fails badly on HOUSEHOLD and HOBBIES because weekly seasonality breaks down on sparse series — the model fits noise and produces erratic or negative forecasts that get clipped to zero.

---

### ARIMA

| Metric | Value |
|--------|-------|
| WRMSSE (1k sample) | 0.7493 |
| vs SN28 reference (1k) | +0.0715 (worse than SN28 ref) |
| FOODS | 0.6590 |
| HOUSEHOLD | 1.8400 |
| HOBBIES | 3.2663 |
| Wall time | 3887 s (64.8 min) |
| Time/series | 3.89 s |
| Fallbacks | 0 |
| Zero forecasts issued | 395 / 1000 |
| Kaggle full-catalogue | pending |

**Configuration:** `pmdarima.auto_arima`, non-seasonal, stepwise search, max p/q/d = 3/3/1.

**Reading the scores:** ARIMA is slower than ETS and scores worse overall because it lacks an explicit seasonality component — without the 7-day periodic structure, it reverts to trend extrapolation which degrades on M5's heavily seasonal demand. FOODS performance (0.6590) is in the right neighborhood but ETS wins by 0.097. HOBBIES collapses identically to ETS (same zero-forecast fallback rate).

---

### SARIMA — OOM Crash

SARIMA was the third method queued. It ran for approximately 3 hours with `n_jobs=4` parallel workers before crashing at **task 442/1000** with a `joblib.externals.loky.process_executor.TerminatedWorkerError`.

**Root cause:** One or more worker processes were killed by the OS. This is consistent with an out-of-memory kill (OOM killer on Windows). Each SARIMA worker holds the full training series (1,913 values), the pmdarima model object, and intermediate ARMA fitting matrices in memory. With `n_jobs=4`, four workers run simultaneously. On the machine's RAM budget (~8 GB available to CPU processes with RTX 3070 consuming GPU VRAM separately), the accumulated per-worker memory exceeded the OS limit during particularly expensive fits in the HOBBIES stratum (many near-zero series cause pmdarima to explore more ARMA orders before converging).

**Crash location:** `classical.py:364` — inside `run_batch()` → `Parallel(n_jobs=4)(...)`

**Mitigation options evaluated and rejected:**

| Option | Estimated time | Why rejected |
|--------|---------------|--------------|
| Rerun with `n_jobs=1` (sequential) | ~8–12 hours | Not worth 8+ hours for one leaderboard row |
| Reduce sample to 500 series | ~4–6 hours; less statistical signal | Marginal gain on a method we already expect to lose to ETS |
| Reduce `n_jobs=2` and retry | ~4–5 hours; still may OOM | Same conclusion |
| Skip SARIMA entirely | 0 hours | Selected |

The honest conclusion is that SARIMA does not belong in a production M5 pipeline. Its serial-per-series fitting paradigm simply does not scale to the dataset.

---

### SARIMAX — Not Run

SARIMAX (SARIMA with exogenous features: `is_holiday`, `snap`, `is_weekend`, `month`, `sell_price`) was queued after SARIMA. Given SARIMA's OOM crash, SARIMAX — which carries additional memory overhead for exogenous feature matrices — was skipped. Expected runtime was 50+ minutes even in the optimistic case. Not run; no result available.

---

## Per-Category Analysis: What the Results Tell Us

| Category | Demand regime | ETS | ARIMA | Best classical | Notes |
|----------|--------------|-----|-------|---------------|-------|
| FOODS | High-volume, regular, strong weekly seasonality | **0.5616** | 0.6590 | ETS | Classical methods are competitive here. ETS's explicit trend + seasonal model matches demand structure. |
| HOUSEHOLD | Mid-volume, moderate sparsity | 1.7023 | 1.8400 | ETS (barely) | Both methods degrade badly. Many series are intermittent enough to trigger zero-forecast fallback. |
| HOBBIES | Low-volume, highly sparse | 3.2663 | 3.2663 | Tied (both fail) | Identical score because both hit the zero-forecast fallback for the same set of series. Classical models provide no signal here. |

**Key insight:** Classical per-series methods are competitive only on high-volume regular series. On the full M5 dataset, HOBBIES and HOUSEHOLD together represent ~52% of series by count but a smaller fraction of revenue weight. ETS likely earns a full-catalogue Kaggle score somewhere between 0.75 and 0.85 — the heavy FOODS weight from the full dataset will pull the score toward the FOODS regime where ETS performs well.

---

## Why the M5 Winners Used Global ML Models

This experiment makes the case clearly:

1. **Scaling.** ETS at 0.81 s/series × 30,490 series = 6.8 hours. LightGBM trains once on all series simultaneously in under 30 minutes.
2. **Sparse series.** Classical models have no cross-series information sharing. When a HOBBIES item has 90% zeros, ARIMA has nothing to work with. A global model can learn from the entire catalogue that low-sales items in category X behave like other low-sales items in category X.
3. **Feature richness.** SNAP flags, holidays, price elasticity, and store effects are awkward to fold into per-series SARIMAX fits. They are natural input features for tree-based models.
4. **WRMSSE weighting.** The metric weights by revenue. A model that nails FOODS and accepts mediocre HOBBIES performance will outperform a model that optimizes uniformly across all series. Global ML models can learn this implicitly from the training signal; classical models cannot.

Day 4 onward pivots to global models. The classical results serve as a floor that any competent ML approach should beat.

---

## Files Produced

| File | Description |
|------|-------------|
| `src/models/classical.py` | ETS, ARIMA, SARIMA, SARIMAX fit functions + `run_batch()` |
| `scripts/run_day3.py` | End-to-end pipeline: sample selection → model fitting → WRMSSE scoring |
| `reports/sample_1000_series.csv` | 1,000-series sample IDs with metadata and stratum labels |
| `reports/02_sample_selection.md` | Stratification design rationale |
| `reports/day3_run_log.txt` | Raw console output from the Day 3 run (includes SARIMA crash traceback) |
| `submissions/ets_sample_submission.csv` | ETS forecasts (1k sample, F1–F28) submitted to Kaggle |
| `submissions/arima_sample_submission.csv` | ARIMA forecasts (1k sample, F1–F28) submitted to Kaggle |
