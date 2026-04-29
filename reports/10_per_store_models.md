# Day 10 — Per-Store LightGBM Models

## Setup

Train one LightGBM model per store (10 models total) with per-store Optuna hyperparameter search, then blend 0.6×per-store + 0.4×global recursive. Hypothesis: store-level models capture demand heterogeneity across CA/TX/WI that a global model must average over.

Training window: d_1000–d_1913 (914 days × 3,049 items = 2,786,786 rows per store).  
Val period: d_1914–d_1941 (single-step, global features parquet).  
Eval period: d_1942–d_1969 (recursive, 28 steps from d_1941 history).  
Optuna: 15 trials per store, Tweedie loss, 500-round early stopping (ES=75, cap=3000).

---

## Results Summary

| Model | Val WRMSSE | Kaggle Public | Kaggle Private |
|-------|-----------|---------------|----------------|
| Global (Day 6, reference) | **0.5422** | 0.5422 | — |
| Per-store only | 0.6140 | 0.6140 | **0.6410** |
| Blend (0.6×per-store + 0.4×global recursive) | 0.5737 | 0.5736 | 0.6430 |

Val WRMSSE: per-store (0.6140) is worse than global (0.5422) by 0.0718 — identical pattern to Day 7, where per-category models (0.5726) also trailed global on val.

---

## Per-Store Optuna Results

| Store | lr | num_leaves | tvp | best_iter | val_tweedie | Optuna time |
|-------|----|-----------|-----|-----------|-------------|-------------|
| CA_1 | 0.100 | 64 | 1.520 | 2999 | 3.876 | 161s |
| CA_2 | 0.025 | 32 | 1.536 | 3000 | 4.046 | 202s |
| CA_3 | 0.025 | 256 | 1.583 | 3000 | 4.443 | 267s |
| CA_4 | 0.100 | 64 | 1.446 | 3000 | 3.102 | 194s |
| TX_1 | 0.025 | 256 | 1.494 | 3000 | 3.176 | 301s |
| TX_2 | 0.075 | 32 | 1.512 | 3000 | 3.595 | 187s |
| TX_3 | 0.100 | 256 | 1.627 | 3000 | 3.371 | 211s |
| WI_1 | 0.075 | 128 | 1.523 | 3000 | 3.633 | 223s |
| WI_2 | 0.100 | 256 | 1.570 | 3000 | 3.678 | 246s |
| WI_3 | 0.100 | 128 | 1.543 | 3000 | 3.371 | 190s |

All 10 stores hit the 3000-round cap without early stopping triggering (ES=75). This indicates underfitting: the models continue to improve monotonically up to the training cap. Increasing `num_boost_round` or lowering lr for stores like CA_2 and CA_3 would likely reduce val loss further.

---

## Cross-Store Demand Heterogeneity

**By state (val tweedie, lower = easier to predict):**

| State | Stores | Val tweedie range | Pattern |
|-------|--------|-------------------|---------|
| CA | CA_1, CA_2, CA_3, CA_4 | 3.10 – 4.44 | Widest spread — heterogeneous |
| TX | TX_1, TX_2, TX_3 | 3.18 – 3.60 | Mid-range, tighter spread |
| WI | WI_1, WI_2, WI_3 | 3.37 – 3.68 | Similar to TX |

**CA_3 is the hardest store (val=4.44):** Highest num_leaves (256) and lowest lr (0.025) — Optuna gravitated toward high-capacity, slow-learning to manage complexity. Despite this, CA_3 still hit the iteration cap, indicating it needs more training.

**CA_4 is the easiest store (val=3.10):** Low-capacity model (64 leaves, lr=0.1) converges fast. CA_4 likely has more regular demand patterns than other CA stores.

**Tweedie power variation reveals demand sparsity differences:**

| Store | tvp | Interpretation |
|-------|-----|----------------|
| TX_3 | 1.627 | Heaviest tail — most compound/intermittent demand in the set |
| CA_3 | 1.583 | Second-highest; volatile mix of FOODS + HOBBIES |
| WI_2 | 1.570 | Significant intermittency |
| CA_4 | 1.446 | Smoothest — near-Poisson regime |
| TX_1 | 1.494 | Near-Poisson; regular TX demand |

tvp range: 1.45–1.63 across stores. The global model used tvp=1.499 (a cross-store average). Per-store models surface that some stores (TX_3, CA_3) need a heavier compound tail while others (CA_4, TX_1) sit closer to Poisson — structural demand heterogeneity that a global tvp cannot simultaneously satisfy.

---

## Val WRMSSE: Why Per-Store is Worse Than Global

Per-store val WRMSSE (0.6140) > global (0.5422) by 0.0718. This is the same pattern as Day 7 (per-category: 0.5726 vs global 0.5422).

**Mechanism:** Each store model trains on 2.79M rows (one store's items × 914 days). The global model trains on 27.9M rows (all stores), giving it 10× more cross-series signal. LightGBM's tree splits on `store_id`, `cat_id`, `dept_id`, `item_id` already partition the space — the global model can learn store-specific patterns through these splits without discarding cross-store transfer.

By training per-store, we lose:
- Cross-store item demand transfer (same SKU sold in CA_1 and TX_2 — correlated demand)
- Cross-state calendar effect calibration (SNAP days affect CA differently than TX)
- 10× smaller training set reduces leaf count reliability at fine partitions

**Why we still expect private LB improvement:** Val WRMSSE measures accuracy on a fixed 28-day window with single-step oracle features. Private LB (recursive, 28 steps forward) is a different regime where prediction diversity across ensemble members reduces correlated error. Day 7 confirmed this: per-category was 0.5726 (worse) on val but contributed −0.101 on private LB via blend. We expect the same mechanism here.

---

## Engineering Notes

**Duplicate `store_id` column bug (fixed before training):**  
Original code listed `store_id` twice — once explicitly, and again as part of `CAT_FEATURES`. This caused pandas to create a duplicate column that broke LightGBM's categorical dtype check. Fixed by removing `store_id` from the explicit column list.

**`free_raw_data=False` (carried from Day 9):**  
Using `reference=ds_tr` in `lgb.Dataset` requires both datasets to keep raw data alive. `free_raw_data=True` caused silent crashes in Day 9's training loop (discovered and fixed then).

**Checkpoint-resumable:**  
`store_{name}.pkl` checked for existence before training. Re-running the script skips already-trained stores. On a crash after CA_3, re-run picks up from CA_4.

---

## Comparison to Day 7 (Per-Category)

| Approach | Val WRMSSE | Val delta vs global | Day 7/10 private LB |
|----------|-----------|---------------------|----------------------|
| Global (Days 6–10 reference) | 0.5422 | — | 0.8138 (Day 7 recursive) |
| Per-category (Day 7) | 0.5726 | +0.0304 | 0.7126 (blend) |
| Per-store (Day 10) | 0.6140 | +0.0718 | pending |
| Per-store blend (Day 10) | 0.5737 | +0.0315 | pending |

Per-store val delta (+0.0718) is larger than per-category (+0.0304), which makes sense: 10-way split is more extreme than 3-way. If the same diversity-vs-accuracy trade-off holds, we'd expect a larger private LB improvement from per-store — but the relationship is not guaranteed to be monotone.

**Private LB reversal — per-store-only beats blend (0.6410 vs 0.6430):**

Blending in 0.4× global recursive *hurts* on the eval period. The global model's recursive forecast (private=0.8138 in isolation) is weaker than per-store recursive (0.641). Adding a weaker component introduces noise rather than complementary signal.

This is the inverse of Day 7: the Day 7 blend (per-category + global) worked because per-category and global had comparable quality on the eval period, so they contributed different error patterns. Here, per-store has already surpassed global — blending in the inferior component is detrimental.

**The val/private discrepancy is structural:** On val (single-step), blend=0.5737 beats per-store-only=0.6140 because the global's single-step predictions are accurate (0.5422) and anchor the blend. On private (recursive), per-store-only=0.6410 beats blend=0.6430 because the global's recursive predictions are weak (0.8138) and degrade the blend.

**Lesson:** Ensembling improves generalisation when components have comparable quality with different error patterns. When one component dominates the other on the eval regime, the weaker component adds correlated noise rather than diversity.
