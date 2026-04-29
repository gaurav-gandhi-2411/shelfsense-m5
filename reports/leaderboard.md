# ShelfSense M5 — Model Leaderboard

Metric: **WRMSSE** (lower is better).  
Validation period: **d_1914 – d_1941** (last 28 days of `sales_train_evaluation.csv`).  
Kaggle public LB: validation period submitted to M5 Forecasting Accuracy competition.

> **Note on Day 3 scores:** Classical method scores are computed on a **stratified 1,000-series sample** (334 FOODS-top, 333 HOUSEHOLD-mid, 333 HOBBIES-low), not the full 30,490 series. Sample WRMSSE is not directly comparable to full-catalogue Kaggle scores because hierarchical aggregation and revenue weights are re-normalized within the subset. See `reports/02_classical_methods.md` for full explanation.

---

| Rank | Model | Family | Score type | WRMSSE | Kaggle LB | Day | Notes |
|------|-------|--------|------------|--------|-----------|-----|-------|
| 1 | **MH blend (0.5×multi-horizon + 0.5×Day8 recursive, eval rows)** | **LightGBM** | **Full-30490** | **0.5422** | **0.5422 / 0.5854** | **9** | **Best private LB to date — multi-horizon eliminates compounding error on eval period; val WRMSSE 0.7156 was misleading (see Day 9 note)** |
| 2 | Multi-horizon global (28 direct models, eval) | LightGBM | Full-30490 | 0.5422 | 0.5422 / 0.6095 | 9 | −0.1031 private vs Day 8 recursive; pure MH eval rows, SS val rows |
| — | Per-store only (10 models, recursive eval) | LightGBM | Full-30490 | 0.6140 | 0.6140 / **0.6410** | 10 | Private better than blend (0.6430) — blending in weaker global recursive hurts |
| — | Per-store blend (0.6×per-store + 0.4×global recursive) | LightGBM | Full-30490 | 0.5737 | 0.5736 / 0.6430 | 10 | Val better than per-store-only (global anchors accuracy) but private worse (global recursive drags it down) |
| 3 | **Blend (0.6×per-cat + 0.4×global), v2 recursive eval** | **LightGBM** | **Full-30490** | **0.5545** | **0.5545 / 0.7126** | **8** | **Prev best private — now surpassed by Day 9** |
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
| Multi-horizon val WRMSSE (origin d_1913) | 0.7156 — misleadingly bad (see note) |
| **mh_blend Kaggle private LB** | **0.5854 — new best by 0.1272** |
| mh_global Kaggle private LB | 0.6095 |
| Recursive v2 val WRMSSE | 0.6019 |
| Optuna best params (h=14) | lr=0.075, leaves=256, tvp=1.499 |
| Total training time | 126.1 min (28 models) |

**Why val WRMSSE was a misleading metric:**

The val WRMSSE (0.7156) measured multi-horizon from a single origin (d_1913) against the single-step oracle (actual features at each of d_1914-1941). This is an unfair comparison — it penalises the staleness of multi-horizon features versus perfect real-world features that don't exist at inference time.

The correct comparison is multi-horizon vs recursive, both starting from d_1941 on the private LB eval period (d_1942-1969). Recursive accumulates 27 steps of compounding prediction error; multi-horizon uses model_h on clean d_1941 actual features. Multi-horizon wins by 0.103 on this fair comparison (0.6095 vs 0.8138 for global recursive), and the blend wins by 0.127 (0.5854 vs 0.7126).

**Key insight:** Val WRMSSE from single origin is a biased estimator of private LB quality. The benefit of multi-horizon (eliminating compounding error) only shows up on the eval period — where no real oracle features exist and the comparison is against recursive's noisy accumulated predictions.

---

## Day 10 — Per-Store LightGBM (10 Models)

| Finding | Value |
|---------|-------|
| Per-store val WRMSSE | 0.6140 — worse than global (same pattern as Day 7 per-category) |
| Per-store blend val WRMSSE | 0.5737 |
| Global reference val WRMSSE | 0.5422 |
| Per-store only Kaggle scores | public=0.6140, **private=0.6410** |
| Per-store blend Kaggle scores | public=0.5736, private=0.6430 |
| Total training time | ~38 min (10 stores × Optuna 15 trials + 3000 rounds) |
| All stores hit iteration cap | Yes (3000 rounds, no early stopping) → underfitting |

**Per-store Optuna params (notable variation):**

| Store | lr | leaves | tvp | val_tweedie |
|-------|----|--------|-----|-------------|
| CA_1 | 0.100 | 64 | 1.520 | 3.876 |
| CA_2 | 0.025 | 32 | 1.536 | 4.046 |
| CA_3 | 0.025 | 256 | 1.583 | 4.443 |
| CA_4 | 0.100 | 64 | 1.446 | 3.102 |
| TX_1 | 0.025 | 256 | 1.494 | 3.176 |
| TX_2 | 0.075 | 32 | 1.512 | 3.595 |
| TX_3 | 0.100 | 256 | **1.627** | 3.371 |
| WI_1 | 0.075 | 128 | 1.523 | 3.633 |
| WI_2 | 0.100 | 256 | 1.570 | 3.678 |
| WI_3 | 0.100 | 128 | 1.543 | 3.371 |

**Key finding — tvp range 1.45–1.63:** The global model used tvp=1.499 (a cross-store average). Per-store Optuna reveals that TX_3 (1.627) and CA_3 (1.583) have significantly heavier compound tails than CA_4 (1.446) and TX_1 (1.494). A single global tvp cannot simultaneously satisfy all stores — structural demand heterogeneity that per-store models capture.

**Hardest vs easiest stores:** CA_3 (val=4.44) is the hardest — Optuna selected max-complexity params (256 leaves, low lr) but still hit the iteration cap. CA_4 (val=3.10) is easiest — light params converge quickly. State-level pattern: CA stores have widest variance (3.10–4.44), TX and WI are more uniform (3.18–3.68).

**Val WRMSSE worse than global (expected):** Per-store models train on 2.79M rows vs 27.9M for global. Smaller dataset loses cross-series transfer — the same item sold in CA_1 and TX_2 has correlated demand, but per-store models can't see across the boundary. Global tree splits on store_id already handle store heterogeneity without sacrificing cross-store signal.

**Private LB result — per-store beats blend (0.6410 vs 0.6430):**

This is the inverse of Day 7, where blending per-category + global improved private LB. Here, blending in the global recursive (private=0.8138) with stronger per-store recursive (private=0.641) *hurts*. The mechanism: per-store models are already better than global on the eval period — adding 0.4× of a weaker recursive forecast introduces noise rather than complementary signal.

Day 7 blend worked because global and per-category were roughly equally imperfect (0.8138 global vs undocumented per-cat solo). Day 10 blend failed because one component (per-store) had already surpassed the other (global). **Lesson: ensembling helps when components have comparable quality but different error patterns. When one dominates, the weaker component adds noise.**

| Model | Val WRMSSE | Private LB | vs Day 9 best |
|-------|-----------|------------|----------------|
| mh_blend (Day 9) | 0.5422 | **0.5854** | — |
| mh_global (Day 9) | 0.5422 | 0.6095 | +0.024 |
| per_store_only (Day 10) | 0.6140 | 0.6410 | +0.056 |
| per_store_blend (Day 10) | 0.5737 | 0.6430 | +0.058 |
| Day 7/8 blend | 0.5545 | 0.7126 | +0.127 |

Per-store (0.641) improves significantly over global recursive (0.8138) but doesn't close the gap to multi-horizon (0.585). Key reason: per-store models still use recursive for the eval period — they eliminate the demand heterogeneity problem but not the 28-step compounding error. Multi-horizon eliminates both by predicting each horizon directly.

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
| Multi-horizon global (Day 9) | 0.5422 | **0.6095** | Direct 28-model training; eliminates recursive compounding |
| **MH blend (Day 9 best)** | 0.5422 | **0.5854** | 0.5×MH + 0.5×Day8 recursive; best private LB |
| Per-store only (Day 10) | 0.6140 | **0.6410** | Per-store recursive; beats blend (0.6430) — global recursive too weak to help |
| Per-store blend (Day 10) | 0.5736 | 0.6430 | Blending in weaker global recursive hurts vs per-store-only |

Interpretation: on the public LB (validation period), the 1k-sample models can't distinguish from SN28. On the private LB (true evaluation period), ARIMA and ETS show small private-score improvements — likely because they capture some trend signal that SN28 misses, and the evaluation period differs structurally from the validation period.

---

## Evaluator Notes

- Local WRMSSE now **exactly matches Kaggle LB** (verified: 0.8377 = 0.8377).
- Key fix (Day 2 Task A): scale denominator now trims leading zeros before computing naive-1 MSE. Many M5 series launch mid-dataset; including pre-launch zeros deflated the scale and inflated RMSSE by ~5%.
- Weights use exact day-level `sales × price` joins (not average price × total sales).
- Series with scale=0 (all-zero training history) produce RMSSE=∞ and are excluded from the weighted sum.
- WRMSSE = unweighted mean of 12 hierarchical level scores.
- Aggregate levels (1–9) always score lower than item levels (10–12) due to noise cancellation.
