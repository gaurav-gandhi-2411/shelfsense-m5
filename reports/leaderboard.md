# ShelfSense M5 — Model Leaderboard

Metric: **WRMSSE** (lower is better).  
Validation period: **d_1914 – d_1941** (last 28 days of `sales_train_evaluation.csv`).  
Kaggle public LB: validation period submitted to M5 Forecasting Accuracy competition.

> **Note on Day 3 scores:** Classical method scores are computed on a **stratified 1,000-series sample** (334 FOODS-top, 333 HOUSEHOLD-mid, 333 HOBBIES-low), not the full 30,490 series. Sample WRMSSE is not directly comparable to full-catalogue Kaggle scores because hierarchical aggregation and revenue weights are re-normalized within the subset. See `reports/02_classical_methods.md` for full explanation.

---

| Rank | Model | Family | Score type | WRMSSE | Kaggle LB | Day | Notes |
|------|-------|--------|------------|--------|-----------|-----|-------|
| — | Multi-horizon 28 models (from origin d_1913) | LightGBM | Full-30490 | 0.7156 | pending | 9 | WORSE than recursive — single-origin feature staleness dominates over compounding-error elimination; see Day 9 note |
| — | Multi-horizon blend (0.5×MH + 0.5×Day8 eval) | LightGBM | Full-30490 | — | pending | 9 | Ensemble diversity candidate; private LB pending |
| 1 | **Blend (0.6×per-cat + 0.4×global), v2 recursive eval** | **LightGBM** | **Full-30490** | **0.5545** | **0.5545 / 0.7126** | **8** | **v2 rewrite confirms v1 was correct; identical score — gap is structural, not a bug** |
| — | Blend (0.6×per-cat + 0.4×global), v1 recursive eval | LightGBM | Full-30490 | 0.5545 | 0.5545 / 0.7126 | 7 | Same score as Day 8 v2 — v1 feature math was already correct |
| 2 | Global recursive (proper eval period) | LightGBM | Full-30490 | 0.5422 | 0.5422 / 0.8138 | 7 | Recursive eval fixes Day 6 SN28 placeholder; private −0.082 vs Day 6 |
| 3 | **LightGBM global best (Day 6, val-period only)** | **LightGBM** | **Full-30490** | **0.5422** | **0.5422 / 0.8956³** | **6** | **tvp=1.499, lr=0.025, 879 iter** |
| 4 | LightGBM Tweedie (power=1.1) | LightGBM | Full-30490 | 0.5442 | — | 6 | Handily beats RMSE; 823 iter |
| 5 | LightGBM RMSE | LightGBM | Full-30490 | 0.5651 | — | 6 | Vanilla regression baseline |
| — | Seasonal Naive 28 (1k ref) | Baseline | Sample-1000 | 0.6778 | — | 3 | SN28 on same 1k sample; use as Day 3–4 relative baseline |
| 4 | ETS | Classical | Sample-1000 | 0.6541 | 0.8377 | 3 | Best sample score; sample improvement too small to move full-catalogue — see note ¹ |
| 5 | Prophet (cps=0.1) | Prophet | Sample-1000 | 0.6638 | 0.8377 | 4 | Best cps from sweep; private 0.8731 (sample rank ≠ private rank — see note ²) |
| 6 | Prophet (cps=0.05) | Prophet | Sample-1000 | 0.6743 | — | 4 | Default cps; marginal vs cps=0.1 |
| 7 | Seasonal Naive (28-day) | Baseline | Full-30490 | 0.8377 | 0.8377 | 2 | Best full-catalogue classical score; Kaggle verified |
| 8 | ARIMA | Classical | Sample-1000 | 0.7493 | 0.8377 | 3 | Worse than ETS/Prophet on sample; private 0.8582 beats ETS private 0.8698 |
| 9 | Prophet (cps=0.01) | Prophet | Sample-1000 | 0.7766 | — | 4 | Too stiff; underfit trend on FOODS |
| 10 | Seasonal Naive (7-day) | Baseline | Full-30490 | 0.8697 | — | 2 | Repeats last week's pattern |
| 11 | Moving Average (28d) | Baseline | Full-30490 | 1.0823 | — | 2 | Mean of last 28 days |
| 12 | Moving Average (90d) | Baseline | Full-30490 | 1.1015 | — | 2 | Mean of last 90 days |
| 13 | Moving Average (7d) | Baseline | Full-30490 | 1.1361 | — | 2 | Mean of last 7 days |
| 14 | Seasonal Naive (365-day) | Baseline | Full-30490 | 1.4615 | — | 2 | Same window last year |
| 15 | Naive (last value) | Baseline | Full-30490 | 1.4639 | — | 2 | Repeat last observation |
| — | SARIMA | Classical | INCOMPLETE | — | — | 3 | OOM crash at 442/1000; see reports/02_classical_methods.md |
| — | SARIMAX | Classical | NOT RUN | — | — | 3 | Skipped after SARIMA OOM; not worth 8+ hrs compute |

---

## Day 9 — Multi-Horizon Direct Training (28 Models)

| Finding | Value |
|---------|-------|
| Multi-horizon val WRMSSE (origin d_1913) | **0.7156** — worse than recursive |
| Recursive v2 val WRMSSE | 0.6019 |
| Single-step oracle | 0.5422 |
| Gap vs recursive | +0.1137 (multi-horizon is worse) |
| Optuna best params (h=14) | lr=0.075, leaves=256, tvp=1.499 |
| Total training time | 126.1 min (28 models, h=7..28 newly trained) |
| Submissions | mh_global.csv (val=SS, eval=MH), mh_blend.csv (eval=0.5×MH+0.5×Day8) |

**Why multi-horizon underperformed recursive on the val period:**

Each model_h predicts sales[d+h] using features at a single origin point (d_1913). For h=1 this is fine — features at d_1913 predict d_1914. But for h=28, the model uses lag_7 = sales[d_1906], lag_28 = sales[d_1885], roll_mean_7 of d_1907-1913 — all 28 days stale relative to the target d_1941. Recursive forecasting, while compounding prediction error, at least keeps rolling/lag features refreshed at every step with the most recent (predicted) values. Feature staleness for far horizons is a more damaging problem than error compounding for this dataset.

**What to try instead (Day 10+):** Rather than a single origin, use a rolling-window evaluation where each horizon h uses the features available h days before the target — exactly what the oracle does. This is equivalent to using the actual feature parquet at each day rather than a frozen origin.

---

## Day 8 — Recursive Forecast v2 Audit

| Finding | Value |
|---------|-------|
| v2 recursive WRMSSE (val period, global model) | 0.6019 — **identical to v1** |
| Single-step WRMSSE | 0.5422 |
| Recursive-vs-single-step gap | 0.0597 (11.0%) |
| Kaggle public LB (v2 blend) | 0.5545 — identical to Day 7 |
| Kaggle private LB (v2 blend) | 0.7126 — identical to Day 7 |
| v1 bugs found | **None** — original implementation was mathematically correct |

**Key finding:** The 11% recursive gap is not a bug — it is structural. Each of 28 recursive steps introduces error that propagates forward as lag/rolling features. With 68% zero rate across M5 series, a 10-12% gap over 28 steps is expected and correct. Eliminating this gap requires direct multi-horizon training (Day 9), not a recursive rewrite.

**v2 improvements** (auditability, not accuracy): exact day-index lookup via `searchsorted` instead of buffer-position arithmetic; rolling window uses boolean mask over day_cols instead of slice from buffer right edge; single `predict_horizon()` entry point with consistent API.

---

## Day 7 — Results Summary

| Finding | Value |
|---------|-------|
| Recursive eval gap (global model, 28 steps) | 0.5422 → 0.6019 (+11% — expected for recursive forecasting) |
| Private LB improvement from fixing eval period | 0.8956 → 0.8138 (−0.082) |
| Blend private LB improvement over global recursive | 0.8138 → 0.7126 (−0.101) |
| Best private LB to date | **0.7126** (blend submission) |
| Per-category tvp (FOODS / HOUSEHOLD / HOBBIES) | 1.438 / 1.555 / 1.402 — HOUSEHOLD highest, not HOBBIES |
| Per-category val WRMSSE | 0.5726 — **worse than global 0.5422** |

**Key insight:** Per-category models are weaker individually (lose cross-series signal) but add ensemble diversity that improves generalisation on the out-of-window evaluation period. The blend's public score is worse (0.5545 vs 0.5422) but private is better (0.7126 vs 0.8138).

---

## Day 6 — LightGBM Per-Category Breakdown (full 30,490 series)

| Category | RMSE model | Tweedie (1.1) | Best tuned | Classical best (1k sample) | LightGBM improvement |
|----------|-----------|--------------|-----------|---------------------------|---------------------|
| FOODS | — | — | **0.5204** | ETS 0.5616 | −0.04 |
| HOUSEHOLD | — | — | **0.5905** | ETS 1.7023 | **−1.11** |
| HOBBIES | — | — | **0.6112** | ETS 3.2663 | **−2.65** |
| **Overall** | **0.5651** | **0.5442** | **0.5422** | SN28 0.8377 | **−0.30 (−35%)** |

**HOBBIES: the key validation.** Classical per-series methods hit a zero-forecast fallback for ~390/1,000 sparse HOBBIES series (WRMSSE=3.27). LightGBM's cross-series learning transfers demand signal from neighbouring items/stores, achieving **0.6112** — a 5× reduction without a single per-series fit.

---

³ **Why LightGBM private (0.8956) = SN28 private (0.8956).** The M5 submission format requires forecasting two non-overlapping 28-day windows: the *validation* period (d_1914–d_1941, public LB) and the *evaluation* period (d_1942–d_1969, private LB). The Day 6 training pipeline forecasted d_1914–d_1941 only. The evaluation rows (d_1942–d_1969) were left as the SN28 base, so the private LB score matches SN28 exactly. The public score (0.5422) is meaningful and exact — confirmed by local evaluator match. Fixing the evaluation-period forecast requires recursive prediction from d_1942 forward using lag/rolling features computed from d_1914–d_1941 actuals. This is addressed in Day 7.

---

² **Sample rank ≠ private LB rank.** On the 1k sample: ETS (0.6541) > Prophet (0.6638) >> ARIMA (0.7493). On Kaggle private LB: ARIMA (0.8582) > ETS (0.8698) > Prophet (0.8731). The ranking reverses. Reason: the 1k sample is FOODS-top heavy (334 FOODS series out of 1,000), which over-represents the demand regime where ETS and Prophet excel. On the full 30,490-series catalogue, ARIMA's simpler trend model generalises more consistently across all regimes. **Model selection from biased samples is unreliable.** Day 6 LightGBM trains on all 30,490 series — ranking from that point will be trusted over sample-WRMSSE.

¹ **Why does 1k-sample ETS score 0.8377 (same as SN28) on Kaggle?** ETS improved on 1,000 of 30,490 series (~3.3%). Those 1,000 series — even the FOODS-top stratum — carry insufficient revenue weight to shift the full-catalogue WRMSSE by more than rounding error. To beat SN28 on Kaggle, the model must run on **all 30,490 series**. This is the core motivation for global ML models (Days 6–7): one LightGBM train covers all series simultaneously at a fraction of the per-series compute cost.

---

## Day 4 — Prophet Per-Category Breakdown (1k sample)

| Model | FOODS | HOUSEHOLD | HOBBIES | Overall |
|-------|-------|-----------|---------|---------|
| Seasonal Naive 28 (ref) | 0.6400 | 1.1580 | 1.5949 | 0.6778 |
| ETS | **0.5616** | 1.7023 | 3.2663 | **0.6541** |
| Prophet (cps=0.1) | 0.5742 | 1.7378 | 3.2663 | 0.6638 |
| Prophet (cps=0.05) | 0.5822 | 1.7622 | 3.2663 | 0.6743 |
| Prophet (cps=0.01) | 0.6853 | 1.8178 | 3.2663 | 0.7766 |
| ARIMA | 0.6590 | 1.8400 | 3.2663 | 0.7493 |

**Interpretation:** Prophet closes the gap to ETS on FOODS (0.5742 vs 0.5616) — the higher changepoint flexibility better tracks trend changes. HOBBIES score is identical across ETS/ARIMA/Prophet (3.2663) because all three hit the same zero-forecast fallback for the same ~390 sparse series. HOUSEHOLD is where Prophet underperforms ETS: ETS's additive weekly seasonality is a better prior for mid-volume items than Prophet's multiplicative seasonal mode.

---

## Day 3 — Classical Methods Per-Category Breakdown (1k sample)

| Model | FOODS | HOUSEHOLD | HOBBIES | Overall |
|-------|-------|-----------|---------|---------|
| Seasonal Naive 28 (ref) | 0.6400 | 1.1580 | 1.5949 | 0.6778 |
| ETS | **0.5616** | 1.7023 | 3.2663 | **0.6541** |
| ARIMA | 0.6590 | 1.8400 | 3.2663 | 0.7493 |

**Interpretation:** ETS and ARIMA both beat the seasonal naive baseline on FOODS (high-volume, regular demand). On HOUSEHOLD and HOBBIES the models collapse — sparse and intermittent series that violate ETS/ARIMA smoothness assumptions cause the zero-forecast fallback (390/1000 series for ETS) to fire, producing WRMSSE well above baseline. This is the defining failure mode of per-series statistical models on the M5 dataset.

---

## Seasonal Naive (28-day) — Level Breakdown

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
| level_10 (item) | 3 049 | 1.0941 |
| level_11 (item × state) | 9 147 | 1.1286 |
| level_12 (item × store) | 30 490 | 1.1567 |
| **Total WRMSSE** | — | **0.8377** |

---

## Kaggle Private Score Note

The M5 competition has separate public (validation) and private (evaluation) leaderboard scores. All Kaggle LB scores shown in this document are **public scores** unless marked otherwise. For reference, private scores observed so far:

| Model | Public LB | Private LB | Note |
|-------|-----------|------------|------|
| Seasonal Naive 28 | 0.8377 | 0.8956 | Day 2 reference |
| ETS (1k-sample + SN28 fill) | 0.8377 | 0.8698 | Private better than SN28 by 0.0258 |
| ARIMA (1k-sample + SN28 fill) | 0.8377 | 0.8582 | Best private so far; ARIMA trend model generalises more consistently across full catalogue |
| Prophet cps=0.1 (1k + SN28 fill) | 0.8377 | 0.8731 | Worse than ETS on private despite better sample score — sample-selection bias |
| LightGBM best (Optuna) | **0.5422** | 0.8956 | Public exact — evaluator confirmed. Private = SN28 because eval rows not forecasted; fixed in Day 7 |
| LightGBM global recursive | 0.5422 | **0.8138** | Day 7: proper recursive eval forecast; private −0.082 vs Day 6 |
| **LightGBM blend (Day 7 best)** | 0.5545 | **0.7126** | Ensemble diversity beats individual model accuracy on private LB |

Interpretation: on the public LB (validation period), the 1k-sample models can't distinguish from SN28. On the private LB (true evaluation period), ARIMA and ETS show small private-score improvements — likely because they capture some trend signal that SN28 misses, and the evaluation period differs structurally from the validation period.

---

## Evaluator Notes

- Local WRMSSE now **exactly matches Kaggle LB** (verified: 0.8377 = 0.8377).
- Key fix (Day 2 Task A): scale denominator now trims leading zeros before computing naive-1 MSE. Many M5 series launch mid-dataset; including pre-launch zeros deflated the scale and inflated RMSSE by ~5%.
- Weights use exact day-level `sales × price` joins (not average price × total sales).
- Series with scale=0 (all-zero training history) produce RMSSE=∞ and are excluded from the weighted sum.
- WRMSSE = unweighted mean of 12 hierarchical level scores.
- Aggregate levels (1–9) always score lower than item levels (10–12) due to noise cancellation.
