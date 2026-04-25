# ShelfSense M5 — Model Leaderboard

Metric: **WRMSSE** (lower is better).  
Validation period: **d_1914 – d_1941** (last 28 days of `sales_train_evaluation.csv`).  
Kaggle public LB: validation period submitted to M5 Forecasting Accuracy competition.

---

| Rank | Model | Family | WRMSSE (local) | Kaggle LB | Day | Notes |
|------|-------|--------|---------------|-----------|-----|-------|
| 1 | Seasonal Naive (28-day) | Baseline | **0.8377** | **0.8377** | 2 | Repeats last 28 days |
| 2 | Seasonal Naive (7-day) | Baseline | 0.8697 | — | 2 | Repeats last week's pattern |
| 3 | Moving Average (28d) | Baseline | 1.0823 | — | 2 | Mean of last 28 days |
| 4 | Moving Average (90d) | Baseline | 1.1015 | — | 2 | Mean of last 90 days |
| 5 | Moving Average (7d) | Baseline | 1.1361 | — | 2 | Mean of last 7 days |
| 6 | Seasonal Naive (365-day) | Baseline | 1.4615 | — | 2 | Same window last year |
| 7 | Naive (last value) | Baseline | 1.4639 | — | 2 | Repeat last observation |

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

## Evaluator Notes

- Local WRMSSE now **exactly matches Kaggle LB** (verified: 0.8377 = 0.8377).
- Key fix (Day 2 Task A): scale denominator now trims leading zeros before computing naive-1 MSE. Many M5 series launch mid-dataset; including pre-launch zeros deflated the scale and inflated RMSSE by ~5%.
- Weights use exact day-level `sales × price` joins (not average price × total sales).
- Series with scale=0 (all-zero training history) produce RMSSE=∞ and are excluded from the weighted sum.
- WRMSSE = unweighted mean of 12 hierarchical level scores.
- Aggregate levels (1–9) always score lower than item levels (10–12) due to noise cancellation.
