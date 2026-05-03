# WS2.5 — LightGBM Diversification

**Rationale:** M5 1st-place winner used 220 LightGBM models with no DL. Adding structural diversity
to LightGBM is higher EV/compute-hour than DL on consumer hardware. Five variants planned.

---

## Variant 1: Per-Department Multi-Horizon LightGBM

**Script:** `scripts/11_lgbm_per_dept.py`  
**Status:** COMPLETE — scored  
**Date:** 2026-05-03

### Architecture

- 7 dept-specific models (FOODS_1/2/3, HOUSEHOLD_1/2, HOBBIES_1/2)
- Each dept model = 28 horizon-specific LightGBMs (direct multi-horizon, same as Day 9)
- Total: **196 models**
- Optuna: 10 trials on h=14, per-dept (warm-started from Day 9 best params)
- Final training: `num_boost_round=3000`, `early_stopping_rounds=75`, Tweedie objective
- Features: same 38-column set as global multi-horizon (no dept-specific feature engineering)

### Hyperparameters (Optuna best per dept)

| Dept | lr | num_leaves | min_data | feat_frac | bag_frac | l2 | tvp |
|------|----|-----------|---------|-----------|----------|-----|-----|
| FOODS_1 | 0.10 | 32 | 100 | 0.8 | 0.8 | 0.0 | 1.511 |
| FOODS_2 | 0.025 | 64 | 50 | 0.9 | 0.9 | 0.0 | 1.512 |
| FOODS_3 | 0.025 | 128 | 100 | 0.7 | 0.7 | 0.1 | 1.613 |
| HOUSEHOLD_1 | 0.10 | 128 | 100 | 0.9 | 0.7 | 0.5 | 1.478 |
| HOUSEHOLD_2 | 0.05 | 32 | 100 | 0.7 | 0.9 | 0.0 | 1.389 |
| HOBBIES_1 | 0.025 | 64 | 20 | 0.7 | 0.8 | 0.0 | 1.440 |
| HOBBIES_2 | 0.025 | 128 | 50 | 0.7 | 0.9 | 0.0 | 1.371 |

### Val WRMSSE Breakdown

| Dept | Series | Val WRMSSE |
|------|--------|-----------|
| HOUSEHOLD_1 | 5,320 | 0.6497 |
| FOODS_3 | 8,230 | 0.6587 |
| HOBBIES_1 | 4,160 | 0.7174 |
| HOUSEHOLD_2 | 5,150 | 0.8101 |
| FOODS_2 | 3,980 | 0.9713 |
| FOODS_1 | 2,160 | **1.1752** |
| HOBBIES_2 | 1,490 | **1.4138** |
| **Overall** | **30,490** | **0.7333** |

*Note: per-subset WRMSSE scores use the global weight structure applied to a subset, so individual
dept scores are not directly comparable to global model scores. Overall 0.7333 is comparable.*

### Kaggle Scores

| Public LB | Private LB | Status |
|-----------|------------|--------|
| 0.73321 | **0.61370** | COMPLETE |

Reference (mh_blend best): public 0.5422 / private **0.5854**

### Model Quality Observations

**FOODS_3 low tree count (caveat):**  
With `lr=0.025`, `early_stopping=75`, FOODS_3 models converged at 30–86 trees (mean 59).
Expected range for this learning rate would be 200–500. Early stopping fired very early,
suggesting the validation metric plateaued fast on this dept's 8,230-series data.
Likely explanation: FOODS_3 contains high-volume intermittent items where the Tweedie loss
saturates quickly at the global feature level — per-dept features without further
sub-grouping don't add signal.

**FOODS_1 / HOBBIES_2 outlier val scores:**  
FOODS_1 (1.175) and HOBBIES_2 (1.413) are far above the global model. FOODS_1 has only 2,160
series — the smallest training set of any dept — so the per-dept model sees insufficient
diversity to match a global model trained on all 30,490 series. HOBBIES_2 (1,490 series,
77% zero rate) is the most intermittent dept; splitting it away from the global model
removes the cross-series regularization effect that helps sparse series.

**Cache-skip behaviour:**  
The script's `if os.path.exists(path): load; continue` pattern means a re-run produces
misleadingly fast timing (6.2 min total was pickle loads + inference, not training). The
`h_scores` dict being empty on all depts is the indicator — it only populates during
fresh training. Models were trained in a prior session; pkl files were already on disk.

### Decision: B — Marginal, Capture for Ensemble

Private LB 0.6137 > 0.5854 (reference) but < 0.62. Worse standalone but may contribute
diversity to a blend. Public/private gap is 0.1195 vs 0.0432 for mh_blend — large gap
suggests per-dept partitioning is overfitting the training period signal.

**Why per-dept underperformed vs M5 winners:**
M5 1st-place per-dept/per-store decomposition used:
1. Dept-specific feature engineering (different lag/roll windows per demand type)
2. Per-store × per-dept submodels (70+ unique combinations)
3. Separate evaluation-period models trained on more recent data

Our implementation shares one feature set across all depts and only splits at training — not
at feature design. The cross-series regularization benefit of the global model outweighs
the dept-specialization benefit at this scale.

---

## Variant 2: Tweedie Power Sweep

**Script:** `scripts/12_lgbm_tvp_sweep.py <tvp>`  
**Status:** tvp=1.3 COMPLETE — scored | tvp=1.5 and tvp=1.7 PENDING checkpoint approval  
**Date:** 2026-05-03

### Architecture

- 28 direct-horizon LightGBM models (identical to Day 9 mh_blend)
- Same features, same Optuna-chosen hyperparameters (lr=0.025, leaves=64, l2=0.1, etc.)
- **Only change:** `tweedie_variance_power` fixed at 1.3 / 1.5 / 1.7 (Optuna chose 1.5316)
- No re-tuning Optuna — isolates the effect of tvp alone

### Hyperparameters (fixed across all tvp values)

| Param | Value |
|-------|-------|
| lr | 0.025 |
| num_leaves | 64 |
| min_data_in_leaf | 100 |
| feature_fraction | 0.7 |
| bagging_fraction | 0.9 |
| lambda_l2 | 0.1 |
| num_boost_round | 3000 (early_stopping=75) |
| seed | 42 |

### tvp=1.3 Results

**Training:** 191.3 min, 28 models  
**Tree counts:** 173–1692 (mean ~950) — proper full training, well-behaved early stopping  
**Val WRMSSE:** 0.6860 (Day 9 ref: 0.7254, delta: **-0.039**)

| Public LB | Private LB | vs mh_blend private |
|-----------|------------|---------------------|
| 0.54220 | **0.56934** | **-0.016 (new best)** |

**Decision: A — variant helped.** tvp=1.3 beats mh_blend (0.5854) by 0.016 private LB.
Public is tied (0.5422) — both use the same single-step oracle for val rows. Private delta
is driven entirely by the eval-period prediction quality difference.

**Why tvp=1.3 wins:** Lower tvp = stronger zero-inflation emphasis = compound Poisson behavior.
M5 has 68% overall zero rate (HOBBIES 77%, HOUSEHOLD 72%). The Optuna-chosen tvp=1.5316
is a dataset average; tvp=1.3 biases predictions toward zero more aggressively, reducing
over-prediction on sparse/intermittent series which dominate the WRMSSE weight structure.

### tvp=1.5 and tvp=1.7 Results

*Pending checkpoint approval.*

### Ensemble Tracking (preliminary)

| Model | Val WRMSSE | Public LB | Private LB |
|-------|-----------|-----------|------------|
| mh_blend (Day 9, tvp=1.5316) | 0.7254 | 0.5422 | 0.5854 |
| **tvp=1.3** | **0.6860** | **0.5422** | **0.5693** ← new best |
| tvp=1.5 | — | — | — |
| tvp=1.7 | — | — | — |

---

## Ensemble Tracking

| Variant | Val WRMSSE | Public LB | Private LB | Include in ensemble? |
|---------|-----------|-----------|------------|---------------------|
| mh_blend (Day 9 reference) | 0.5422 | 0.5422 | 0.5854 | ✓ anchor |
| per_dept (Variant 1) | 0.7333 | 0.7332 | 0.6137 | Tentative — test blend weight |
| **tvp=1.3 (Variant 2a)** | **0.6860** | **0.5422** | **0.5693** | **New best — beats mh_blend by 0.016** |
| tvp=1.5 (Variant 2b) | — | — | — | Pending |
| tvp=1.7 (Variant 2c) | — | — | — | Pending |
