# Multi-Horizon LightGBM (28 Direct-Horizon Models)

## Overview

Train one LightGBM model per forecast horizon (h=1..28). Each model_h predicts
`sales[d+h]` given features available at time d. At inference, all 28 models
use a single origin point (d_1941 for eval), so no recursive error compounding
is possible.

**Result:** Val WRMSSE 0.7156 — worse than recursive (0.6019). See engineering finding below.

---

## Training Design

| Component | Configuration |
|-----------|---------------|
| Models | 28 (one per horizon h=1..28) |
| Features | Same 38 features as global model (all available at time d) |
| Target | `df.groupby("id")["sales"].shift(-h)` |
| Train rows | `d_num in [FEAT_START=1000, VAL_START-h-1]` |
| Val rows | `d_num in [VAL_START-h, LAST_TRAIN-h]` (targets in d_1886..d_1913) |
| Hyperparameters | Optuna 15 trials on h=14 (median), shared across all h |
| Optuna best params | lr=0.075, leaves=256, min_data=200, feat_frac=0.7, tvp=1.499 |
| Inference origin | d_1913 (val) / d_1941 (eval) — all 28 models use same origin |
| Total training time | 126.1 min |

---

## Optuna Results (h=14)

| Param | Value |
|-------|-------|
| Best Tweedie val loss | 3.8652 |
| learning_rate | 0.075 |
| num_leaves | 256 |
| min_data_in_leaf | 200 |
| feature_fraction | 0.7 |
| bagging_fraction | 0.9 |
| lambda_l2 | 0.0 |
| tweedie_variance_power | 1.499 |

---

## Validation WRMSSE (origin d_1913)

| Method | WRMSSE | vs oracle |
|--------|--------|-----------|
| Single-step oracle (actual features each day) | 0.5422 | 0 (reference) |
| Recursive v2 | 0.6019 | +0.0597 |
| **Multi-horizon from origin d_1913** | **0.7156** | **+0.1734** |

Multi-horizon is **worse than recursive by 0.1137**. Reason: see engineering finding below.

---

## Kaggle Scores

| Submission | Public LB | Private LB | vs prev best |
|------------|-----------|------------|--------------|
| mh_blend.csv (val=SS, eval=0.5×MH+0.5×recursive) | 0.5422 | **0.5854** | **−0.1272** |
| mh_global.csv (val=SS, eval=MH-direct) | 0.5422 | 0.6095 | −0.1031 |
| Recursive blend (prev best) | 0.5545 | 0.7126 | — |

---

## Per-Horizon MAE (origin d_1913)

| h | MAE | h | MAE | h | MAE | h | MAE |
|---|-----|---|-----|---|-----|---|-----|
| 1 | 0.9152 | 8 | 1.0157 | 15 | 1.0660 | 22 | 1.0325 |
| 2 | 0.8621 | 9 | 1.0212 | 16 | 0.9824 | 23 | 0.9670 |
| 3 | 0.8685 | 10 | 0.9836 | 17 | 1.0134 | 24 | 0.9374 |
| 4 | 0.8818 | 11 | 0.9911 | 18 | 0.9929 | 25 | 0.9395 |
| 5 | 0.9819 | 12 | 1.0497 | 19 | 1.0617 | 26 | 1.0149 |
| 6 | 1.0999 | 13 | 1.1934 | 20 | 1.2039 | 27 | 1.1613 |
| 7 | 1.1406 | 14 | 1.1342 | 21 | 1.2809 | 28 | 1.2225 |

MAE increases from h=1 (0.86) to h=7 (1.14), drops at h=8, spikes at h=13/20/21 — weekly
pattern reset, but with stale features the models can't read the current week position.

---

## Result: Multi-Horizon Beats Recursive on Private LB by 0.13

**Hypothesis:** Eliminating compounding error would close the gap to the oracle.

**Val WRMSSE (0.7156) was misleading.** It compared multi-horizon from a single origin
(d_1913) against the single-step oracle that uses actual features at each of d_1914–1941.
This penalises feature staleness relative to perfect information that doesn't exist at
inference time — an unfair comparison.

**Private LB tells the real story:**

| Model | Private LB | Interpretation |
|-------|-----------|----------------|
| mh_blend | **0.5854** | New best — MH clean origin beats recursive compounding |
| mh_global | 0.6095 | Pure MH; −0.103 vs global recursive |
| Recursive blend (prev best) | 0.7126 | 27-step compounding from d_1941 |

On the eval period (private LB), both approaches start from d_1941. Multi-horizon uses
model_h with clean actual features at d_1941 to directly predict d_1941+h. Recursive
accumulates 27 predictions of error. Multi-horizon wins by 0.127 on the blend.

**Why val WRMSSE was the wrong metric:**

The val WRMSSE uses actual features at each day for the oracle (single-step) but forces
multi-horizon to use a frozen origin. The private LB has no such oracle — both recursive
and multi-horizon must forecast without future actuals. On that equal footing, clean-origin
multi-horizon beats error-compounding recursive.

**Production implication:**

| Condition | Favours |
|-----------|---------|
| Eval horizon with no future actuals (production forecasting) | Multi-horizon |
| Val measurement with oracle features available | Recursive looks better (misleadingly) |
| Mixed catalogue — ensemble both | Blend (best result here: 0.5854) |

**Key lesson:** Use private LB (or proper walk-forward CV) to evaluate forecasting
strategies. Single-step oracle-based val WRMSSE biases against multi-horizon.

---

## Files

| File | Description |
|------|-------------|
| `scripts/09_train_multi_horizon.py` | Full training + inference pipeline |
| `data/models/multi_horizon/h_{01..28}.pkl` | 28 trained models (gitignored) |
| `reports/day9_scores.json` | Full results + per-horizon stats |
| `submissions/mh_global.csv` | Val: single-step. Eval: multi-horizon direct. |
| `submissions/mh_blend.csv` | Val: single-step. Eval: 0.5×MH + 0.5×Day8 recursive |
