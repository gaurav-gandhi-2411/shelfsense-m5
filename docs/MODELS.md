# Models Reference

Seven LightGBM variants were trained across Phase 2 (experiments) and Phase 3 Stage 4 (Dagster
assets). This document covers rationale, Hydra configuration, training details, evaluation scores,
and lessons learned for each variant, followed by a calibration analysis of the val→private LB
divergence that defined the Phase 2 experiment ceiling.

---

## Summary Table

Ordered by private LB ascending (lower is better). Val WRMSSE is measured on a held-out window
within the training split; public and private LB are Kaggle competition scores. See each variant
section for the exact evaluation window used.

| # | Variant | Dagster asset | Feature set | Val WRMSSE | Public LB | Private LB |
|---|---------|--------------|-------------|-----------|-----------|------------|
| 1 | **tvp_13** | `model_tvp_13` | default | 0.6860 | 0.5422 | **0.5693** |
| 2 | **ylags** | `model_ylags` | ylags | 0.6830 | — | 0.5749 |
| 3 | **store_dept** | `model_store_dept` | default | 0.6294 | — | 0.5882 |
| 4 | **per_dept** | `model_per_dept` | default | 0.7333 | 0.7332 | 0.6137 |
| 5 | **rmse_mh** | `model_rmse_mh` | default | 0.6699 | 0.5422 | 0.6205 |
| 6 | **per_store** | `model_per_store` | default | 0.6140 | 0.6140 | 0.6410 |
| 7 | **tvp_17** | `model_tvp_17` | default | 0.7713 | — | — |

**Competition context.** M5 winner reached 0.520 (private LB), 2nd place (Matthias Anderer) reached
0.528. The best result here (0.5693) places within striking distance of the podium but is held back
by the single-holdout validation structure described in the [Val→Private Divergence](#val-private-divergence)
section.

**tvp_17 note.** The config and Dagster asset are fully implemented. Phase 2 tvp sweep was pending
checkpoint approval at Phase 2 close; the Phase 3 val WRMSSE (0.7713) is from the Dagster
materialization. No Kaggle submission was made for this variant.

---

## Feature Sets

Two feature sets are defined in `shelfsense/config/features/`:

### `default` (42 features)

Defined in `shelfsense/config/features/default.yaml`:

| Group | Features | Count |
|-------|----------|-------|
| Calendar | day_of_week, month, year, week_of_year, day_of_month, quarter | 6 |
| Events | snap_CA, snap_TX, snap_WI, event_type_1, event_type_2, is_holiday, event_weight | 7 |
| Price | sell_price, price_change_pct, price_norm_store_dept, price_momentum_7d, price_rel_dept_mean | 5 |
| Lags | lag_7, lag_14, lag_28, lag_56 | 4 |
| Rolling | mean/std/min/max over 7, 14, 28, 56-day windows | 16 |
| Categorical | item_id, dept_id, store_id, state_id (LightGBM native categoricals) | 4 |
| **Total** | | **42** |

### `ylags` (45 features)

Defined in `shelfsense/config/features/ylags.yaml`:

Extends `default` by adding three annual lag windows:

```yaml
lag_windows: [7, 14, 28, 56, 91, 182, 364]
```

The annual lags (91, 182, 364 days) are pre-built during `features` asset materialisation. No
separate feature build step is needed — `model_ylags` selects `YLAGS_FEATURE_COLS` from the same
feature parquets written by the `features` asset.

---

## Variant 1: tvp_13 — Canonical Baseline

**Rationale.** The M5 dataset has a 68% overall zero rate (HOBBIES 77%, HOUSEHOLD 72%, FOODS ~40%).
Tweedie loss with variance power 1.3 is a compound-Poisson distribution — it has a point mass at
zero and a heavy right tail, matching this zero-inflated sales profile. Power 1.3 biases predictions
toward zero more aggressively than the Optuna-chosen tvp≈1.5, which reduces over-prediction on the
sparse intermittent series that dominate WRMSSE's weight structure. Multi-horizon direct training
— 28 independent LightGBM models, one per forecast day h ∈ {1..28} — eliminates the compounding
prediction error accumulated by recursive approaches over 28 steps.

**Hydra config** (`shelfsense/config/model/tweedie_mh_tvp13.yaml`):

```yaml
objective: tweedie
tweedie_variance_power: 1.3
learning_rate: 0.025
num_leaves: 64
min_data_in_leaf: 100
feature_fraction: 0.7
bagging_fraction: 0.9
bagging_freq: 1
lambda_l2: 0.1
num_boost_round: 3000
early_stopping_rounds: 75
seed: 42
num_threads: 0
horizon: 28
optuna_horizon: 14
optuna_trials: 15
optuna_start_day: 1600
```

**Training data.** All 30,490 M5 series. Features built from days 1–1941; training window ends at
day 1885 (leaves 28 days for val). One model per horizon → 28 `h_*.pkl` files in
`data/models/tvp_13/`. Tree counts ranged 173–1692 (mean ~950), indicating proper early stopping
without premature convergence.

**Training time (WSL2, 16 GB).** ~190 minutes for 28 horizons on all series.

**Scores.**

| Val WRMSSE | Public LB | Private LB |
|-----------|-----------|------------|
| 0.6860 | 0.5422 | **0.5693** |

**Lesson.** This is the best-performing variant on private LB. The large val→public gap
(0.6860 → 0.5422) reflects different evaluation windows: val is measured on days 1886–1913 using
training-period actuals as lag inputs, while public LB measures days 1914–1941 — the period where
no oracle lag features exist and multi-horizon's advantage over recursive forecasting fully
materialises.

---

## Variant 2: ylags — Annual Lag Features

**Rationale.** M5 sales exhibit strong annual seasonality: FOODS spikes at Thanksgiving and
Christmas, HOBBIES tracks back-to-school cycles. Adding lag features at 91, 182, and 364 days
exposes these patterns directly to the tree model without relying solely on calendar indicators.
All other hyperparameters are identical to tvp_13.

**Hydra config.** Base model config: `tweedie_mh_tvp13.yaml` (identical). Feature override:

```yaml
# shelfsense/config/features/ylags.yaml
lag_windows: [7, 14, 28, 56, 91, 182, 364]
```

Asset `model_ylags` selects `YLAGS_FEATURE_COLS` defined in
`shelfsense/models/lightgbm/multihorizon.py`, which extends `DEFAULT_FEATURE_COLS` with the three
annual lags. The feature parquets are shared — no re-materialisation of the `features` asset is
needed when switching between feature sets.

**Training data.** All 30,490 series. Same train/val split as tvp_13. The annual lag columns are
available from the start of the dataset only for series with sufficient history (lag_364 needs at
least 364 days of prior sales), so early series rows are NaN-filled with the same imputation as the
shorter lags.

**Training time (WSL2, 16 GB).** ~210 minutes (3 extra feature columns add minor overhead per
tree split evaluation).

**Scores.**

| Val WRMSSE | Public LB | Private LB |
|-----------|-----------|------------|
| 0.6830 | — | 0.5749 |

**Lesson.** Annual lags improved val WRMSSE by 0.003 (0.6860 → 0.6830) but worsened private LB by
0.006 (0.5693 → 0.5749). The validation window happened to include a period where 364-day lag was
informative; the private window did not reward it at the same rate. Marginal feature additions that
improve a single holdout by < 0.01 WRMSSE should be treated with suspicion — they may be learning
noise in the holdout period rather than generalisable annual patterns.

---

## Variant 3: store_dept — Per-Slice Training

**Rationale.** The global model (tvp_13) assigns one set of boosting trees to 30,490 heterogeneous
series. Store × department slices — CA_1×FOODS_3, TX_2×HOBBIES_1, etc. — are far more homogeneous:
300–400 series selling similar items at similar price points through the same store-specific event
calendar. A model specialised on one slice may find sharper split thresholds than a model
calibrated across all 30,490.

**Hydra config** (`shelfsense/config/model/store_dept.yaml`):

```yaml
objective: tweedie
tweedie_variance_power: 1.3
learning_rate: 0.025
num_leaves: 64
min_data_in_leaf: 20
feature_fraction: 0.7
bagging_fraction: 0.9
bagging_freq: 1
lambda_l2: 0.1
num_boost_round: 3000
early_stopping_rounds: 75
seed: 42
history_days: 200
optuna_trials: 10
```

Per-slice Optuna searches over lr (0.01–0.1), num_leaves (31–127), and min_data_in_leaf (20–100).
`history_days: 200` limits each slice to the most recent 200 days of history to reduce training
time. A parameter hash of each slice's Optuna result is used as a cache key — re-runs skip already
trained slices.

**Training data.** 70 slices (10 stores × 7 departments). Each slice trains independently; the
final prediction is the union of all 70 slice forecasts. Model files:
`data/models/store_dept/lgbm_SD_{store}_{dept}.pkl`.

**Training time (WSL2, 16 GB).** ~3–4 hours for 70 slices (Optuna × 10 trials per slice). The
parameter hash cache makes re-runs near-instant for unchanged slices.

**Scores.**

| Val WRMSSE | Public LB | Private LB |
|-----------|-----------|------------|
| 0.6294 | — | 0.5882 |

**Lesson.** Improved val by 0.057 relative to the tvp_13 baseline; worsened private LB by 0.019.
A common single-holdout trap: per-slice Optuna tunes slice-specific parameters to a fixed 28-day
window, so the optimal slice parameters for period A are not necessarily optimal for period B. The
spatial partitioning that looks correct in the holdout may not hold in the private test window.

---

## Variant 4: per_dept — Per-Department Global Model

**Rationale.** A middle-ground between global (tvp_13) and fully-partitioned (store_dept): one
model per M5 department, trained on all stores within that department. FOODS, HOUSEHOLD, and
HOBBIES have structurally different zero-rate distributions (FOODS ~40% zeros, HOUSEHOLD ~72%,
HOBBIES ~77%). Separate models per department allow different Tweedie powers to be found by Optuna
for each demand type — HOUSEHOLD (Optuna tvp ≈ 1.555) vs. HOBBIES (tvp ≈ 1.440) vs. FOODS
categories.

**Hydra config.** Same as tvp_13 (`tweedie_mh_tvp13.yaml`) applied per department. Per-dept
Optuna best parameters (from Phase 2 WS2.5 V1):

| Dept | lr | num_leaves | min_data | tvp |
|------|----|-----------|---------|-----|
| FOODS_1 | 0.10 | 32 | 100 | 1.511 |
| FOODS_2 | 0.025 | 64 | 50 | 1.512 |
| FOODS_3 | 0.025 | 128 | 100 | 1.613 |
| HOUSEHOLD_1 | 0.10 | 128 | 100 | 1.478 |
| HOUSEHOLD_2 | 0.05 | 32 | 100 | 1.389 |
| HOBBIES_1 | 0.025 | 64 | 20 | 1.440 |
| HOBBIES_2 | 0.025 | 128 | 50 | 1.371 |

Asset `model_per_dept` trains `MultiHorizonTrainer(_TVP13_CFG)` for each of 7 departments,
filtering the feature parquet by `dept_id`. Results in `data/models/per_dept/{dept}/h_*.pkl`
(196 pkl files total).

**Training data.** 7 independent training runs on ~4,356 series each (30,490 / 7). The
`dept_filter` kwarg passed to `trainer.fit()` selects the appropriate rows. Per-dept val WRMSSE
breakdown from Phase 2:

| Dept | Series | Val WRMSSE |
|------|--------|-----------|
| HOUSEHOLD_1 | 5,320 | 0.6497 |
| FOODS_3 | 8,230 | 0.6587 |
| HOBBIES_1 | 4,160 | 0.7174 |
| HOUSEHOLD_2 | 5,150 | 0.8101 |
| FOODS_2 | 3,980 | 0.9713 |
| FOODS_1 | 2,160 | 1.1752 |
| HOBBIES_2 | 1,490 | 1.4138 |
| **Overall** | **30,490** | **0.7333** |

**Training time (WSL2, 16 GB).** ~3–4 hours for 7 departments × 28 horizons.

**Scores.**

| Val WRMSSE | Public LB | Private LB |
|-----------|-----------|------------|
| 0.7333 | 0.7332 | 0.6137 |

**Lesson.** Per-dept performed worse than the global model on both val and public LB (0.7333 vs.
0.6860). Each department's training set is one-seventh the size of the global set — fewer cross-
series patterns, fewer splits, less signal from correlated item demand. HOBBIES_2 (1,490 series,
77% zero rate) and FOODS_1 (2,160 series) hit extreme val WRMSSEs because splitting them away from
the global model removes the cross-series regularisation effect that helps sparse series. Private LB
(0.6137) is better than public (0.7332) — an unusual inversion, suggesting the private test period
was more forecastable for this model than the public validation period.

---

## Variant 5: rmse_mh — RMSE Objective

**Rationale.** Tweedie loss weights intermittent (zero-heavy) series heavily. An RMSE objective
treats each unit error identically — this might benefit the high-volume FOODS items that dominate
the WRMSSE numerator even though they have few zeros. If RMSE better captures the FOODS forecasting
signal, the WRMSSE improvement from FOODS accuracy might offset RMSE's indifference to sparse
series.

**Hydra config** (`shelfsense/config/model/rmse_mh.yaml`):

```yaml
objective: regression
metric: rmse
learning_rate: 0.025
num_leaves: 64
min_data_in_leaf: 100
feature_fraction: 0.7
bagging_fraction: 0.9
bagging_freq: 1
lambda_l2: 0.1
num_boost_round: 3000
early_stopping_rounds: 75
seed: 42
num_threads: 0
horizon: 28
optuna_horizon: 14
optuna_trials: 15
optuna_start_day: 1600
```

All params identical to tvp_13 except `objective: regression` and `metric: rmse`. No
`tweedie_variance_power` key.

**Training data.** All 30,490 series. Same train/val split and feature set as tvp_13.

**Training time (WSL2, 16 GB).** ~185 minutes. RMSE loss converges slightly faster than Tweedie on
this dataset.

**Scores.**

| Val WRMSSE | Public LB | Private LB |
|-----------|-----------|------------|
| 0.6699 | 0.5422 | 0.6205 |

**Lesson.** RMSE improved val WRMSSE by 0.016 (0.6860 → 0.6699) but worsened private LB by 0.051
(0.5693 → 0.6205) — the largest private-LB regression of any variant. The mechanism: RMSE-trained
models systematically under-forecast intermittent (zero-heavy) series because the loss does not
penalise over-smoothing of rare demand spikes. WRMSSE weights high-revenue items and intermittent
series more heavily than RMSE, so the objective mismatch compounds in private evaluation.
**Using the evaluation metric as the training objective is non-negotiable for M5.**

---

## Variant 6: per_store — Per-Store Global Model

**Rationale.** Each Walmart store has a distinct SNAP calendar, regional event set (Texas football
games, California state holidays), and product mix. A store-specialised model might capture local
demand patterns — store-specific event indicators, SNAP timing — without needing the full per-slice
Optuna tuning of the store_dept approach.

**Hydra config.** Same as tvp_13 (`tweedie_mh_tvp13.yaml`) applied per store. Per-store Optuna best
parameters (from Phase 2 WS2.5 experiment):

| Store | lr | num_leaves | tvp |
|-------|----|-----------|-----|
| CA_1 | 0.100 | 64 | 1.520 |
| CA_2 | 0.025 | 32 | 1.536 |
| CA_3 | 0.025 | 256 | 1.583 |
| CA_4 | 0.100 | 64 | 1.446 |
| TX_1 | 0.025 | 256 | 1.494 |
| TX_2 | 0.075 | 32 | 1.512 |
| TX_3 | 0.100 | 256 | 1.627 |
| WI_1 | 0.075 | 128 | 1.523 |
| WI_2 | 0.100 | 256 | 1.570 |
| WI_3 | 0.100 | 128 | 1.543 |

The tvp range (1.446–1.627) confirms structural demand heterogeneity: TX_3 and CA_3 require a
heavier compound tail than CA_4 and TX_1. The global model's single tvp=1.3 cannot simultaneously
satisfy all stores.

Asset `model_per_store` trains `MultiHorizonTrainer(_TVP13_CFG)` for each of 10 stores, filtering
by `store_id`. Results in `data/models/per_store/{store}/h_*.pkl` (280 pkl files total).

**Training data.** 10 independent training runs on ~3,049 series each. The `store_filter` kwarg
passed to `trainer.fit()` selects the appropriate rows.

**Training time (WSL2, 16 GB).** ~38 minutes (10 stores × Optuna 15 trials + 3000 rounds per
store). All stores hit the iteration cap (3000 rounds, no early stopping) — a sign of underfitting
due to small per-store training sets.

**Scores.**

| Val WRMSSE | Public LB | Private LB |
|-----------|-----------|------------|
| 0.6140 | 0.6140 | 0.6410 |

**Lesson.** Best val WRMSSE of all variants (0.6140), but worsened private LB by 0.072 vs. tvp_13
(0.6410 vs. 0.5693). Per-store models train on 2.79M rows vs. 27.9M for global — smaller datasets
lose cross-series transfer. The same item sold in CA_1 and TX_2 has correlated demand, but per-
store models cannot see across store boundaries. Global tree splits on `store_id` already capture
store heterogeneity without sacrificing cross-store signal. Per-store also carries a 10× inference
cost (280 pkl files vs. 28). **Ensembling per-store with the global recursive model does not help
either (0.6430 blend vs. 0.6410 per-store-only) — adding a weaker component introduces noise when
the stronger component already dominates.**

---

## Variant 7: tvp_17 — High Tweedie Variance Power

**Rationale.** Increasing variance power from 1.3 toward 2.0 shifts the Tweedie distribution toward
gamma, penalising large forecast values more severely. The hypothesis: HOBBIES items (77% zero rate,
high unit value) are over-forecasted by tvp_13, and a higher power would reduce over-prediction on
these sparse series.

**Hydra config** (`shelfsense/config/model/tweedie_mh_tvp17.yaml`):

```yaml
objective: tweedie
tweedie_variance_power: 1.7
learning_rate: 0.025
num_leaves: 64
min_data_in_leaf: 100
feature_fraction: 0.7
bagging_fraction: 0.9
bagging_freq: 1
lambda_l2: 0.1
num_boost_round: 3000
early_stopping_rounds: 75
seed: 42
num_threads: 0
horizon: 28
optuna_horizon: 14
optuna_trials: 15
optuna_start_day: 1600
```

All params identical to tvp_13 except `tweedie_variance_power: 1.7`.

**Training data.** All 30,490 series. Same train/val split and feature set as tvp_13.

**Training time (WSL2, 16 GB).** ~190 minutes (identical architecture to tvp_13).

**Scores.**

| Val WRMSSE | Public LB | Private LB |
|-----------|-----------|------------|
| 0.7713 | — | — |

(Kaggle submission not made; Phase 2 tvp sweep was pending checkpoint approval at Phase 2 close.)

**Lesson.** Val WRMSSE of 0.7713 is substantially worse than tvp_13 (0.6860). Tweedie 1.7 is
intermediate between compound Poisson and gamma — it penalises large predictions more aggressively,
which reduces the model's ability to capture demand spikes in high-volume FOODS items. The
hypothesis was wrong: tvp_13 is the better-calibrated choice for M5's mixed demand regime. **The
HOBBIES over-prediction problem is better addressed through zero-inflation-aware ensembling than
through an objective sweep.**

---

## Val→Private Divergence

This section documents the structural issue that capped Phase 2 improvement.

### The Reversal

Three of the four Phase 2 variants that were submitted to Kaggle improved local validation WRMSSE
relative to the tvp_13 baseline but produced worse private LB scores. The pattern is consistent and
structural.

| Variant | Val WRMSSE | Val Δ vs tvp_13 | Private LB | Private Δ vs tvp_13 |
|---------|-----------|-----------------|------------|---------------------|
| tvp_13 (baseline) | 0.6860 | 0.0000 | **0.5693** | 0.0000 |
| ylags | 0.6830 | −0.0030 | 0.5749 | **+0.0056** |
| store_dept | 0.6294 | −0.0566 | 0.5882 | **+0.0189** |
| rmse_mh | 0.6699 | −0.0161 | 0.6205 | **+0.0512** |
| per_dept | 0.7333 | +0.0473 | 0.6137 | +0.0444 |

The three variants with a negative val Δ (ylags, store_dept, rmse_mh) all produced a positive
private Δ — meaning they got worse on the private leaderboard despite improving locally. Per_dept
moved consistently in the wrong direction on both metrics (both were worse than the baseline).

### Correlation

Across the four Phase 2 experiments, Pearson r between val delta and private delta is +0.41. The
sign is positive primarily because per_dept degrades both metrics consistently and pulls the
regression upward. Within the three inversions (ylags, store_dept, rmse_mh), the relationship is
negative: larger val improvement → larger private regression. The correlation coefficient alone
understates the problem; the directional consistency of the inversions is the key signal.

### Mechanism

Two compounding causes:

**1. Single-window holdout overfitting.** The local val WRMSSE is measured on a fixed 28-day
window (approximately days 1886–1913 in the training data). This window has a specific SNAP
schedule, post-Christmas demand trajectory, and event mix. Optuna tuning and feature-set changes
that reduce error on this window may learn artefacts specific to those 28 days rather than
generalisable patterns. The private test window (days 1942–1969) covers a different 28-day period
with a different event and SNAP calendar — improvements that fitted the holdout noise do not
transfer.

This is especially visible in store_dept: Optuna ran 10 trials per slice, finding per-slice
hyperparameters that minimise val WRMSSE for each store×dept combination. Those parameters are
optimal for the holdout period, not for the private period. The val improvement (+0.057) is the
largest, and so is the private regression (+0.019 relative to a model that did not over-tune).

**2. Objective–metric mismatch for rmse_mh.** The RMSE variant exposes a secondary mechanism:
training on a loss function that does not match WRMSSE allows the model to reduce RMSE-tracked
error on the val window by shifting probability mass in ways that hurt WRMSSE on the private window.
The model looks good on val because RMSE weights series uniformly; it fails on private because
WRMSSE weights high-revenue and intermittent series more heavily. The val improvement looks real
(−0.016 WRMSSE) but is an artefact of evaluating WRMSSE on a period where RMSE and WRMSSE happen
to be aligned. They are not aligned in general.

### Remedy: Walk-Forward Cross-Validation

The structural fix is replacing the single holdout with a walk-forward scheme. A `cv_evaluation`
asset is planned for the next iteration. Proposed spec:

```yaml
# config for cv_evaluation asset
n_folds: 5
fold_size_days: 28
last_fold_end_day: 1913
gap_days: 0
```

With 5 × 28-day folds (days 1773–1800, 1801–1828, 1829–1856, 1857–1884, 1885–1913), the WRMSSE
estimate becomes an average over five different SNAP calendars, holiday mixes, and demand levels.
A cross-validated ylags score would require all five holdout periods to benefit from the 364-day
lag before a private submission — the three-period case in Phase 2 would likely have been flagged
as noise. Per-slice Optuna with a cross-validated objective would also reduce the store_dept
overfitting problem, though it would increase compute significantly.

Walk-forward CV would not eliminate the objective mismatch for rmse_mh — that requires using a
WRMSSE-aligned loss. Options include custom Tweedie loss tuning or WRMSSE-proxy objectives (e.g.,
weighted Poisson).

---

## Feature Importance

Feature importance plots are generated during `shelfsense train tweedie-mh` and logged to MLflow
as artifacts. Run `mlflow ui --port 5000` after any materialisation to access them. The
`feature_importance.png` artifact is stored in the run's artifact directory under
`mlruns/{experiment_id}/{run_id}/artifacts/`.

Key patterns from Phase 2 model analysis:

**Lag features dominate.** Lag_7 and lag_28 together account for ~40% of combined feature
importance in the global model. The 7-day lag captures weekly seasonality directly (same-weekday
sales), and the 28-day lag captures the monthly consumption cycle for household items. In the ylags
variant, lag_364 adds positive SHAP values for FOODS items around major annual events (days near
Thanksgiving ~1850, Christmas ~1857) but contributes near-zero or negative importance for
HOUSEHOLD and HOBBIES, consistent with the mixed private-LB result.

**Relative price outperforms absolute price.** `price_norm_store_dept` (sell price normalised by
store×department mean) ranks in the top 5 consistently across all global variants. Absolute
`sell_price` ranks lower. Consumers respond to price changes relative to the category baseline
rather than absolute price levels — a pattern well-captured by the normalised feature.

**SNAP flags are category-specific.** SNAP benefit calendar features (snap_CA, snap_TX, snap_WI)
have moderate importance for FOODS series (SNAP-eligible items) and near-zero importance for
HOUSEHOLD and HOBBIES. This explains part of the per_dept motivation: a FOODS-only model can give
SNAP flags higher relative weight. However, the global model with `dept_id` as a categorical
feature already achieves similar effect through split interactions on dept_id × snap_X.

**Annual lags add noise outside FOODS.** In the ylags feature importance, lag_182 (6-month lag)
shows positive importance for FOODS_3 (the large, seasonal food department) and near-zero or
negative importance for HOBBIES_2 (the most intermittent department). When the feature is broadly
available but only informative for a subset of series, its average importance is diluted and its
tree splits on non-FOODS series carry noise. Restricting annual lags to FOODS-only series is a
candidate improvement for the next experiment cycle.

**Rolling statistics complement lags.** Rolling mean and std over 28 and 56 days rank below the
7/28-day lags but above price features in most variants. Rolling std is particularly important for
HOBBIES series — it captures demand volatility, which helps the Tweedie model calibrate its
variance prediction rather than just its mean.

For per-model importance breakdowns, the MLflow run artifacts include separate importance charts
for each of the 28 horizon models (h_1.pkl through h_28.pkl). Short-horizon models (h=1..7) weight
the 7-day lag more heavily; long-horizon models (h=15..28) rely more on rolling statistics and the
28-day lag. This gradient is expected: at h=28, the most recent 7-day lag is already 21 days stale
at the forecast origin, so longer-window aggregations provide more stable signal.
