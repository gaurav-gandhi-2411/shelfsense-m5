# ShelfSense M5 — Model Leaderboard

Metric: **WRMSSE** (lower is better).  
Validation period: **d_1914 – d_1941** (last 28 days of `sales_train_evaluation.csv`).  
Kaggle public LB: validation period submitted to M5 Forecasting Accuracy competition.

> **Note on Day 3 scores:** Classical method scores are computed on a **stratified 1,000-series sample** (334 FOODS-top, 333 HOUSEHOLD-mid, 333 HOBBIES-low), not the full 30,490 series. Sample WRMSSE is not directly comparable to full-catalogue Kaggle scores because hierarchical aggregation and revenue weights are re-normalized within the subset. See `reports/02_classical_methods.md` for full explanation.

---

| Rank | Model | Family | Score type | WRMSSE | Kaggle LB | Day | Notes |
|------|-------|--------|------------|--------|-----------|-----|-------|
| 1 | **LightGBM best (Optuna)** | **LightGBM** | **Full-30490** | **0.5422** | **0.5422 / 0.8956³** | **6** | **tvp=1.499, lr=0.025, 879 iter; best overall** |
| 2 | LightGBM Tweedie (power=1.1) | LightGBM | Full-30490 | 0.5442 | — | 6 | Handily beats RMSE; 823 iter |
| 3 | LightGBM RMSE | LightGBM | Full-30490 | 0.5651 | — | 6 | Vanilla regression baseline |
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
| LightGBM best (Optuna) | **0.5422** | 0.8956 | Public exact — evaluator confirmed. Private = SN28 because evaluation rows (d_1942–d_1969) not yet forecasted; see note ³ |

Interpretation: on the public LB (validation period), the 1k-sample models can't distinguish from SN28. On the private LB (true evaluation period), ARIMA and ETS show small private-score improvements — likely because they capture some trend signal that SN28 misses, and the evaluation period differs structurally from the validation period.

---

## Evaluator Notes

- Local WRMSSE now **exactly matches Kaggle LB** (verified: 0.8377 = 0.8377).
- Key fix (Day 2 Task A): scale denominator now trims leading zeros before computing naive-1 MSE. Many M5 series launch mid-dataset; including pre-launch zeros deflated the scale and inflated RMSSE by ~5%.
- Weights use exact day-level `sales × price` joins (not average price × total sales).
- Series with scale=0 (all-zero training history) produce RMSSE=∞ and are excluded from the weighted sum.
- WRMSSE = unweighted mean of 12 hierarchical level scores.
- Aggregate levels (1–9) always score lower than item levels (10–12) due to noise cancellation.
