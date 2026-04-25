# EDA Findings — M5 Walmart Sales Forecasting

**Date:** 2026-04-25  
**Dataset:** M5 Forecasting Accuracy (Kaggle)

---

## 1. Dataset Summary

| Metric | Value |
|---|---|
| Total time series | 30,490 |
| History length | 1,941 days (~5.3 years) |
| Date range | 2011-01-29 → 2016-06-19 |
| Unique items | 3,049 |
| Stores | 10 (CA×4, TX×3, WI×3) |
| States | 3 (CA, TX, WI) |
| Departments | 7 |
| Categories | 3 (FOODS, HOBBIES, HOUSEHOLD) |
| Forecast horizon | 28 days |

---

## 2. Sales Distribution

- **Total historical units sold:** ~143M across all series
- **Overall zero-sales rate: 68.0%** — most days, most items sell nothing
- Zero rate by category:
  - FOODS: **61.8%** zeros (least sparse)
  - HOUSEHOLD: **71.6%** zeros
  - HOBBIES: **77.1%** zeros (most sparse)
- Right-skewed sales: median daily per series = 0, mean ~0.5
- Top items are almost entirely in FOODS_3 department

**Demand type classification:**
| Type | Criteria | Count | % |
|---|---|---|---|
| Smooth | <5% zeros | 184 | 0.6% |
| Low intermittent | 5–30% zeros | 2,967 | 9.7% |
| Intermittent | 30–70% zeros | 10,615 | 34.8% |
| Lumpy/Erratic | >70% zeros | 16,724 | 54.8% |

**Implication:** Over half of all series are lumpy/erratic. Naive models, LightGBM with Tweedie loss, and zero-inflated approaches are essential. Classical ARIMA is ill-suited to this zero-inflation pattern.

---

## 3. Seasonality

### Weekly
- Saturday is the single highest-sales day; Sunday second.
- Monday is the lowest. Weekend lift is ~15–20% above Monday.

### Monthly
- December spike visible (Christmas shopping + stocking up).
- February dip (short month + post-holiday low).
- Relatively modest month-to-month variation outside of December.

### Yearly
- Consistent upward trend 2011→2015, slight flattening 2015→2016.
- Year-over-year growth ~5–8% in total sales.

### Holiday Impact
- **Christmas day itself**: negative (stores closed / traffic low).
- Pre-Christmas week is the true sales peak (not captured in single-day event flags).
- Easter and Thanksgiving show moderate lifts.
- Sports events (SuperBowl) show minimal aggregate impact.

### SNAP Impact
- **SNAP days lift total sales by +11.2%**.
- FOODS-only SNAP lift: **+15.0%** — confirms that EBT-eligible food categories drive this effect.
- SNAP affects CA, TX, WI independently; highest CA SNAP overlap with food-dominant stores.
- **Implication:** SNAP flags are among the highest-value calendar features for FOODS models.

---

## 4. Price-Sales Relationship

- Price range: ~$0.50 (basic food items) → ~$20+ (electronics/hobby items).
- Scatter shows expected downward slope (higher price → lower volume) but weak at aggregate level.
- **Promotion weeks** (price drop >10%): only **0.25%** of all item-week observations.
  - Promotions are rare but may have outsized sales lifts when they occur.
  - Price features should include: current price, price vs. item mean, week-over-week price change.
- HOBBIES items have higher price variance but lower volume sensitivity (inelastic relative to FOODS).

---

## 5. Hierarchical Structure

- **CA dominates sales** (~44% of total), followed by TX (~33%), WI (~23%).
- Within CA: CA_3 and CA_1 are the highest-volume stores.
- Store-level variance is significant — a single global model must include store embeddings.
- Department-level patterns are highly distinct:
  - FOODS_3 is the single largest department by volume.
  - HOBBIES_2 has the lowest and sparsest sales.
  - HOUSEHOLD_1 and HOUSEHOLD_2 are intermediate.
- **Implication:** Top-down aggregation loses information at item level. Bottom-up forecasting with hierarchy-aware features is preferred.

---

## 6. Time Series Characteristics

- **ADF stationarity test (100 sampled series):** 90% are stationary at p<0.05.
  - FOODS series tend toward stationarity around an upward trend.
  - HOBBIES series more likely non-stationary (sporadic demand).
- ACF for top FOODS item shows strong **weekly seasonality** (lag-7 spike), with secondary lags at 14 and 21.
- PACF shows significant partial autocorrelations at lags 1 and 7, suggesting AR(1) + seasonal AR(7) components.
- Trend decomposition shows clear upward trend with strong weekly seasonal component.
- **Implication:** Lag-7 and lag-28 features are critical. Seasonal differencing at period=7 helps ARIMA-family models.

---

## 7. Modeling Implications Summary

| Insight | Impact |
|---|---|
| 68% zeros overall | Use Tweedie or Poisson loss in LightGBM; zero-inflated models |
| Weekly seasonality dominant | Include lag-7, lag-14, lag-28 features; SARIMA(p,d,q)(P,D,Q)7 |
| SNAP +11–15% lift | SNAP flags are high-value features, especially for FOODS |
| Only 0.6% smooth series | Classical per-series ARIMA is impractical for full dataset |
| HOBBIES 77% zeros | Separate model or separate loss for HOBBIES may help |
| CA dominates (44%) | Store embeddings essential; CA models will drive overall WRMSSE |
| Price promotions rare (0.25%) | Price change features matter only for the small promo subset |
| 90% stationary (ADF) | Minimal differencing needed; direct forecasting viable |

---

## 8. Figures Generated

| File | Description |
|---|---|
| `fig_01_hierarchy.png` | Items per category, series per store, series per dept |
| `fig_02_sales_distribution.png` | Total daily sales, histogram, zero-fraction, category breakdown |
| `fig_03_seasonality.png` | DOW, monthly, yearly, holiday impact, SNAP |
| `fig_04_price_sales.png` | Price-sales scatter, log-log by category, promotion histogram |
| `fig_05_hierarchical.png` | State, store, department time series |
| `fig_06_acf_pacf.png` | ACF/PACF for top FOODS item |
| `fig_07_decomposition.png` | Seasonal decomposition (trend + seasonal + residual) |
| `fig_08_demand_types.png` | Demand type pie, stationarity by category |
