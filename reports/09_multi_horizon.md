# Day 9 — Multi-Horizon LightGBM (28 Direct-Horizon Models)

## Overview

Train one LightGBM model per forecast horizon (h=1..28). Each model_h predicts
`sales[d+h]` given features available at time d. At inference, all 28 models
use a single origin point (d_1941 for eval), so no recursive
error compounding is possible.

---

## Training Design

| Component | Configuration |
|-----------|---------------|
| Models | 28 (one per horizon h=1..28) |
| Features | Same 38 features as Day 6 global (all available at time d) |
| Target | `df.groupby("id")["sales"].shift(-h)` |
| Train rows | `d_num in [FEAT_START, VAL_START - h - 1]` |
| Val rows | `d_num in [VAL_START - h, LAST_TRAIN - h]` (targets in d_1886..d_1913) |
| Hyperparameters | Optuna 15 trials on h=14, shared across all h |
| Optuna best params | lr=0.025 leaves=64 tvp=1.532 |
| Inference origin | d_1941 (all 28 models, actual sales features, no recursion) |

---

## Optuna Results (h=14)

- Best Tweedie val loss: 3.8434
- tvp: 1.532
- lr: 0.025  num_leaves: 64

---

## Validation WRMSSE (origin d_1913)

| Method | WRMSSE | Gap vs oracle |
|--------|--------|---------------|
| Single-step oracle (Day 6/7, actual features each day) | 0.5422 | 0 (reference) |
| **Multi-horizon from origin d_1913** | **0.7254** | **+0.1832** |
| Recursive v2 (Day 8) | 0.6019 | +0.0597 |

Multi-horizon val WRMSSE lies between oracle (uses actual features per day) and
recursive (compounds prediction error over 28 steps). The gap vs oracle remains
because features at d_1913 are 1-28 days stale for predicting d_1914..d_1941.

---

## Per-Horizon MAE

| h | MAE | h | MAE | h | MAE | h | MAE |
|---|-----|---|-----|---|-----|---|-----|
| 1 | 0.9152 | 8 | 1.0154 | 15 | 1.0774 | 22 | 1.0362 |
| 2 | 0.8621 | 9 | 1.0236 | 16 | 0.9826 | 23 | 0.9761 |
| 3 | 0.8685 | 10 | 0.9887 | 17 | 1.0164 | 24 | 0.9374 |
| 4 | 0.8818 | 11 | 0.9911 | 18 | 0.9929 | 25 | 0.9437 |
| 5 | 0.9819 | 12 | 1.0504 | 19 | 1.0647 | 26 | 1.0226 |
| 6 | 1.0999 | 13 | 1.2037 | 20 | 1.2128 | 27 | 1.1753 |
| 7 | 1.1352 | 14 | 1.1342 | 21 | 1.3015 | 28 | 1.2124 |

---

## Key Insight: Why Multi-Horizon Should Win on Private LB

The private LB period (d_1942–d_1969) requires forecasting 28 days from d_1941.

- **Recursive**: d_1969 prediction uses 27 compounded predicted lag features (noisy)
- **Multi-horizon**: d_1969 prediction uses model_28 on d_1941's actual features (clean)

For HOBBIES (77% zeros), recursive compounding causes near-zero predictions to dominate lag
features, leading to persistent under-prediction. Multi-horizon model_28 was trained
specifically on 28-step-ahead patterns and has seen many (d, d+28) training pairs,
learning the typical 28-day-ahead demand distribution directly.

---

## Files

| File | Description |
|------|-------------|
| `scripts/09_train_multi_horizon.py` | Full training + inference pipeline |
| `data/models/multi_horizon/h_{01..28}.pkl` | 28 trained models (gitignored) |
| `reports/day9_scores.json` | Full results + per-horizon stats |
| `submissions/mh_global.csv` | Val: single-step. Eval: multi-horizon global. Submit this. |
| `submissions/mh_blend.csv` | Val: single-step. Eval: 0.5×MH + 0.5×Day8 blend |
