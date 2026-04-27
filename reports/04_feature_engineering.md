# Day 5 — Feature Engineering Pipeline

## Overview

Day 5 builds the feature dataset that all ML models (LightGBM Days 6–7, LSTM/TFT Days 8–9) will consume. The pipeline converts the wide-format M5 sales matrix into a long-format feature table: one row per (series, day), ~48 feature columns, written to partitioned parquet.

**Output:** `data/processed/features/` — 10 parquet files (one per store), readable together with `pd.read_parquet("data/processed/features/")`.

---

## Feature List and Rationale

### Lag Features (4 features)

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `lag_7` | sales[d-7] | Captures same-weekday last week; strongest single predictor in M5 (lag-7 ACF dominant from EDA) |
| `lag_14` | sales[d-14] | Two-week lag; reinforces weekly seasonality signal |
| `lag_28` | sales[d-28] | Same day 4 weeks ago; captures 4-week promotional cycles |
| `lag_56` | sales[d-56] | 8-week lag; captures medium-term trend shifts |

**Leakage:** All lags use `groupby(id).shift(N)` — lag at day d = sales at day d-N. No same-day or future values are used.

**M5 winner alignment:** Yeon-jik Yang (#1 solution) used lag 7, 28, 364 as primary lag features. Our set is slightly more conservative (no 364-day lag because many series have <2 years of non-zero history, making the annual lag noisy).

---

### Rolling Statistics (16 features)

4 statistics × 4 windows = 16 features. All computed with `shift(1)` applied before rolling — the window [d-w-1 .. d-1] is used, so same-day sales never appear in the window.

| Window | Mean | Std | Min | Max | Rationale |
|--------|------|-----|-----|-----|-----------|
| 7 days | ✓ | ✓ | ✓ | ✓ | Short-term level and recent volatility |
| 28 days | ✓ | ✓ | ✓ | ✓ | Monthly smoothing — removes day-of-week noise |
| 56 days | ✓ | ✓ | ✓ | ✓ | Medium-term trend signal |
| 180 days | ✓ | ✓ | ✓ | ✓ | Long-term baseline level; captures seasonal drift |

`roll_mean_7` is the most predictive rolling feature for point forecast accuracy. `roll_std_*` features help LightGBM learn demand volatility — high std → Tweedie/Poisson loss is more appropriate. `roll_min_*` and `roll_max_*` encode range information that tree models use effectively for clipping anomalies.

---

### Calendar Features (13 features)

| Feature | Type | Note |
|---------|------|------|
| `weekday` | int8 (0=Mon..6=Sun) | Primary seasonality driver; Saturday = peak in M5 EDA |
| `month` | int8 | Captures seasonal patterns (holiday shopping months) |
| `quarter` | int8 | Coarser temporal grouping; useful for tree splits |
| `year` | int16 | Trend direction; LightGBM can use as ordinal |
| `day_of_month` | int8 | Month-start/end effects (paycheck cycles) |
| `week_of_year` | int8 | 52-week seasonality signal without a 365-day lag |
| `is_weekend` | int8 | Binary; Sat/Sun have systematically higher M5 sales |
| `is_holiday` | int8 | M5 event_name_1 or event_name_2 is non-null |
| `is_snap_ca` | int8 | SNAP program day for California |
| `is_snap_tx` | int8 | SNAP program day for Texas |
| `is_snap_wi` | int8 | SNAP program day for Wisconsin |
| `days_since_event` | float32 | Days elapsed since last calendar event; captures pre/post event decay |
| `days_until_next_event` | float32 | Days until next event; captures anticipatory demand lift |

**SNAP:** From Day 1 EDA — SNAP days lift FOODS sales +15%. Using separate state flags (not a single `is_snap`) because the program cycles are state-specific and don't correlate across states.

**days_until/since_event:** Widely used in M5 winning solutions to capture the halo effect around events (e.g., Thanksgiving demand lift begins 3–4 days before). Computed efficiently with `np.searchsorted` over sorted event day indices.

---

### Price Features (5 features)

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `sell_price` | raw weekly price | Primary price signal; needed for revenue-weighted loss calibration |
| `price_change_pct` | (price_t − price_{t−1}) / price_{t−1} | Price promotion detection; promotions spike demand |
| `price_relative_mean` | sell_price / item-store historical mean | Normalised price signal; tells model if item is currently cheap or expensive relative to its own history |
| `price_volatility` | rolling std of price over 28 weeks | Items with high price volatility respond more strongly to promotions |
| `has_price_change` | \|price_change_pct\| > 1% | Binary flag for non-trivial price changes; cleaner split point for tree models than continuous change_pct |

**Leakage:** price_change_pct and price_volatility use `shift(1)` before rolling — current week's price is included in `sell_price` (it's known at forecast time: prices are released before the week begins in M5). The M5 competition provides sell prices for the forecast horizon, so this is valid.

---

### Hierarchy Features (4 features, categorical dtype)

| Feature | Cardinality | Note |
|---------|-------------|------|
| `cat_id` | 3 | FOODS, HOUSEHOLD, HOBBIES |
| `dept_id` | 7 | FOODS_1/2/3, HOUSEHOLD_1/2, HOBBIES_1/2 |
| `store_id` | 10 | CA/TX/WI stores |
| `state_id` | 3 | CA, TX, WI |

**LightGBM categorical support:** All four are cast to `pd.CategoricalDtype` with fixed, globally consistent category lists. LightGBM's `categorical_feature` parameter uses these directly — no one-hot encoding needed. Tree splits on categorical features use an optimal grouping algorithm (exponential in cardinality, feasible for ≤10 categories).

**Why not item_id?** Item_id has 3,049 unique values — too high cardinality for LightGBM's categorical split algorithm at default settings. Instead, item-level patterns are captured by the lag and rolling features (which are per-series by construction).

---

## Memory and Compute Strategy

### Why batch by store?

| Batch scope | Rows | Peak memory estimate |
|-------------|------|---------------------|
| Full dataset | 59.2M | ~8.3 GB |
| Per category | 10–28M | 1.5–4 GB |
| **Per store** | **~5.9M** | **~0.9 GB** |

Processing the full dataset at once exceeds the 8 GB RAM budget (with GPU and OS overhead). Per-store batching keeps each batch under 1 GB, runs cleanly, and produces 10 parquet files that can be read back in full (59.2M rows, ~3–5 GB parquet) or lazily filtered by store for training.

### float32 throughout

All numerical features use `float32` (not `float64`). This halves memory during construction and halves I/O at training time. Precision loss is negligible — M5 sales values are integers, prices have at most 2 decimal places, and ratios are bounded.

### Parquet + Snappy

Output format: Snappy-compressed parquet. Snappy provides ~3× compression on float32 data with very fast decompression — appropriate for a training dataset that will be read multiple times per experiment.

---

## M5 Winners' Feature Comparison

| Feature group | This pipeline | Yeon-jik Yang (#1) | Uber Michelangelo approach |
|---------------|--------------|--------------------|-----------------------------|
| Lag features | 4 (7/14/28/56) | 7/28/364 + recursive | lag-7/14/28/35/42 |
| Rolling stats | 4 windows × 4 stats | Mean only, many windows | Mean + std |
| Calendar | Full (13 cols) | Full | Full |
| Price | 5 features | 5–7 features | Price + discount flag |
| Hierarchy cats | 4 (cat/dept/store/state) | Same + item embedding | Label encoded |
| Total features | ~48 | ~50–60 | ~40–50 |

Key difference: winning solutions often added **recursive lag features** (lag of rolling mean, lag of lag) and **cross-series statistics** (mean sales in the same store on the same day). These are intentionally excluded here to keep the pipeline interpretable and fast — Day 6 LightGBM experiments will determine if additional features are worth the complexity.

---

## Sanity Checks

Four checks run automatically at end of `scripts/05_build_features.py`:

1. **Row count:** 30,490 × 1,941 = 59,181,690 rows expected.
2. **Categorical dtypes:** `cat_id`, `dept_id`, `store_id`, `state_id` must have `category` dtype.
3. **NaN after d_num=180:** No NaN expected in lag or rolling features (min_periods=1 for rolling; lags have NaN only for d_num < lag which is before d_180).
4. **Lag correctness spot-check:** For 3 random series, `lag_7[d=100]` must equal `sales[d=93]`.

---

## Output Files

| File | Rows | Note |
|------|------|------|
| `data/processed/features/store_CA_1.parquet` | ~5.9M | |
| `data/processed/features/store_CA_2.parquet` | ~5.9M | |
| `data/processed/features/store_CA_3.parquet` | ~5.9M | |
| `data/processed/features/store_CA_4.parquet` | ~5.9M | |
| `data/processed/features/store_TX_1.parquet` | ~5.9M | |
| `data/processed/features/store_TX_2.parquet` | ~5.9M | |
| `data/processed/features/store_TX_3.parquet` | ~5.9M | |
| `data/processed/features/store_WI_1.parquet` | ~5.9M | |
| `data/processed/features/store_WI_2.parquet` | ~5.9M | |
| `data/processed/features/store_WI_3.parquet` | ~5.9M | |
| **Total** | **~59.2M** | Data directory gitignored; regenerate with `python scripts/05_build_features.py` |

The parquet directory is gitignored (`data/processed/`). To reproduce: run `scripts/05_build_features.py` with the raw M5 CSVs in `data/raw/m5-forecasting-accuracy/`. Total generation time: ~20–35 min on an 8-core machine.
