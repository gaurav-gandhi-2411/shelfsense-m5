# PLAN — Project 4: M5 Forecasting Showcase

## Project Overview

**Goal:** Build a comprehensive forecasting pipeline on the M5 Walmart sales dataset that demonstrates progression from classical statistical methods through modern ML to deep learning, with rigorous evaluation and a polished portfolio presentation.

**Why this project:** Showcases production forecasting expertise (built on real FedEx work), covers the full method spectrum interviewers ask about, and produces a quantifiable leaderboard result on a recognized public benchmark (M5 competition).

**Audience:** Interviewers for Senior Data Scientist / Applied ML roles, especially in retail / supply chain / forecasting-heavy domains (Walmart, Amazon, Flipkart, Uber, Swiggy, etc.).

**Time budget:** 10 days, ~3-4 hours/day = 30-40 hours.

**Compute strategy:**
- Local RTX 3070 8GB: development, EDA, classical methods, small ML experiments
- Kaggle Notebooks (free P100 16GB, 30 hrs/week): full training runs for ML/DL models

---

## Dataset Overview

**M5 Walmart Sales:**
- 30,490 daily time series (item × store)
- 1,941 days of history (~5.3 years)
- 3,049 unique items × 10 stores (3 states: CA, TX, WI)
- Hierarchical structure: item → dept → category → store → state
- Calendar with holidays, SNAP events, prices over time
- Forecast horizon: 28 days
- Metric: Weighted Root Mean Squared Scaled Error (WRMSSE)
- Public leaderboard (validation period): d_1914-d_1941
- Private leaderboard (evaluation period): d_1942-d_1969

**Why this dataset is interview gold:**
- Real retail forecasting at scale
- Hierarchical aggregation challenges
- Intermittent demand (many zeros, especially HOBBIES category)
- Seasonality at multiple levels (weekly, monthly, yearly, holiday)
- Price effects, promotions, calendar events
- Sparse sales data — production-realistic

---

## Method Coverage

This project will systematically implement and evaluate **8 forecasting families**, ordered from simple to complex:

| # | Method | Family | Why include |
|---|---|---|---|
| 1 | Naive / Seasonal Naive | Baseline | Floor for any meaningful forecast |
| 2 | Exponential Smoothing (ETS) | Classical | Production-standard for retail |
| 3 | ARIMA / SARIMA | Classical | Your FedEx experience baseline |
| 4 | SARIMAX | Classical + exog | Adds price, holidays, SNAP |
| 5 | Prophet | Decomposition | Industry-standard, recruiter-recognized |
| 6 | LightGBM (single global model) | Tree-based ML | M5 winners used this |
| 7 | LSTM / GRU per-series | Deep learning | Sequence modeling demonstration |
| 8 | Temporal Fusion Transformer (TFT) | Modern DL | SOTA, attention-based, interpretable |

**Optional stretch (if time permits):**
- 9. N-BEATS / N-HiTS (recent neural methods)
- 10. Hierarchical reconciliation (MinT, OLS) — bottoms-up forecasts that respect hierarchy

---

## 10-Day Plan

### Day 1: Setup, EDA, and Baselines

**Morning (2 hrs):**
- Set up GitHub repo: `m5-forecasting-showcase`
- Folder structure: `data/`, `notebooks/`, `src/`, `models/`, `reports/`, `tests/`
- Set up Kaggle API for dataset download
- Install dependencies: pandas, statsmodels, prophet, lightgbm, pytorch, pytorch-forecasting, sktime, neuralprophet
- Configure local + Kaggle environment

**Afternoon (2 hrs):**
- Comprehensive EDA notebook (`notebooks/01_eda.ipynb`):
  - Data structure, hierarchies, missing values
  - Sales distribution, zero-inflation analysis
  - Seasonality decomposition (weekly, monthly, yearly)
  - Price-sales relationship
  - Holiday/event impact analysis
  - Category/department comparison (FOODS vs HOBBIES vs HOUSEHOLD)
- Save EDA insights to `reports/01_eda_findings.md`

**Deliverable:**
- Repo set up
- EDA notebook with 15-20 visualizations
- Insights document

**Commit:** `feat: initial setup and EDA notebook`

---

### Day 2: Naive Baselines + Evaluation Framework

**Morning (2 hrs):**
- Implement WRMSSE evaluator from scratch (`src/evaluation/wrmsse.py`)
- Verify against M5 reference implementation
- Build leaderboard tracker (`reports/leaderboard.md`) — append every model result here

**Afternoon (2 hrs):**
- Naive baselines (`src/models/naive.py`):
  - Naive (last value)
  - Seasonal Naive (28-day, 7-day, 365-day)
  - Moving average (7d, 28d, 90d)
- Run all baselines on validation period
- Generate baseline submission file
- Submit to Kaggle, record public LB score

**Deliverable:**
- WRMSSE implementation tested
- 6 baseline scores on leaderboard

**Commit:** `feat: WRMSSE evaluator + naive baselines`

---

### Day 3: Classical Statistical Methods

**Morning (3 hrs):**
- Exponential Smoothing (ETS) on a sample of 1,000 series:
  - Auto-select trend/seasonality components
  - Use statsmodels' `ExponentialSmoothing`
  - Parallelize with joblib
- Save predictions, compute WRMSSE

**Afternoon (3 hrs):**
- ARIMA / SARIMA per-series for same 1,000 sample:
  - Auto-ARIMA via pmdarima for order selection
  - SARIMA with weekly seasonality
- SARIMAX with exogenous variables:
  - Add: price, holiday flags, SNAP flags, weekday/weekend
- Compare ETS vs ARIMA vs SARIMAX results

**Note:** 1,000 series sample is intentional — running ARIMA on all 30,490 series locally would take ~24+ hours. Document this trade-off in README.

**Deliverable:**
- ETS, ARIMA, SARIMA, SARIMAX scores on leaderboard
- Per-series timing analysis

**Commit:** `feat: classical forecasting methods (ETS, ARIMA, SARIMAX)`

---

### Day 4: Prophet + Hierarchy Exploration

**Morning (2 hrs):**
- Prophet baseline on sample series:
  - Add holidays, custom seasonalities
  - Tune changepoint sensitivity
- Compare against ARIMA-family
- Discuss when Prophet wins vs loses (great writeup material)

**Afternoon (2 hrs):**
- Hierarchical aggregation analysis:
  - Aggregate to category × state level (more stable signal)
  - Top-down forecast disaggregation
  - Compare bottom-up vs top-down errors
- Create visualizations of error by hierarchy level

**Deliverable:**
- Prophet score on leaderboard
- Hierarchical aggregation findings document

**Commit:** `feat: Prophet model + hierarchical analysis`

---

### Day 5: Feature Engineering for ML Models

**Full day (4 hrs):**

Build the feature pipeline that ML/DL models will use (`src/features/`):

- **Lag features:** 7, 14, 28, 56 day lags
- **Rolling stats:** mean, std, min, max over 7/28/56/180 days
- **Calendar features:** weekday, month, quarter, year, day-of-month, week-of-year
- **Event features:** holiday indicators, SNAP indicators, days-to-next-event
- **Price features:** price change ratio, price relative to average, price volatility
- **Hierarchy features:** category, department, store, state encodings
- **Item-level stats:** item lifetime, days since first sale, sales velocity

Save engineered features as parquet for reuse across ML models.

**Deliverable:**
- Feature engineering module
- Engineered training matrix on disk
- Memory-efficient using float32 + categorical types

**Commit:** `feat: comprehensive feature engineering pipeline`

---

### Day 6-7: LightGBM (Global Model)

**Day 6 — Training:**

Train a single global LightGBM model on all series:
- Use Kaggle Notebook (P100 GPU not strictly needed — LightGBM is CPU)
- Use the engineered features from Day 5
- Optimize for RMSE loss (matches WRMSSE objective)
- Cross-validation strategy: time-based, last 28 days as val
- Hyperparameter tuning with Optuna (50 trials)

**Day 7 — Iteration:**
- Try Tweedie loss (handles zero-inflation common in retail)
- Try Poisson loss for count data
- Compare different feature subsets
- Final model: best loss × best features
- Submit to Kaggle public leaderboard

**Deliverable:**
- LightGBM model with WRMSSE score
- Hyperparameter optimization log
- Feature importance plot

**Commit:** `feat: LightGBM global model with Tweedie loss`

---

### Day 8: Deep Learning — LSTM / GRU

**Morning (2 hrs):**
- Implement LSTM forecaster using PyTorch:
  - Sequence-to-sequence architecture
  - Encoder: 28 days history → hidden state
  - Decoder: 28 days forecast
  - Train on Kaggle GPU
- Train on 5,000 series sample

**Afternoon (2 hrs):**
- GRU variant
- Compare with LSTM
- Add categorical embeddings (item_id, store_id, dept_id)
- Generate predictions, score

**Deliverable:**
- LSTM and GRU models scored
- Discussion of why DL underperforms or outperforms LightGBM (M5 consistently shows LGBM wins for tabular)

**Commit:** `feat: LSTM/GRU sequence models`

---

### Day 9: Temporal Fusion Transformer

**Full day (4 hrs):**

Implement TFT using `pytorch-forecasting`:
- Set up DataLoader with categoricals + time-varying features
- Configure attention over past + future
- Train on Kaggle GPU (this is where free GPU time matters most)
- Generate predictions
- Visualize attention weights for interview-ready interpretability story

**Deliverable:**
- TFT model with WRMSSE score
- Attention visualization for sample predictions
- Comparison vs all other methods

**Commit:** `feat: Temporal Fusion Transformer with attention visualization`

---

### Day 10: Ensembling + README + Polish

**Morning (2 hrs):**
- Build weighted ensemble of best 3-4 models
- Try simple averaging, weighted averaging, stacking via linear regression
- Submit ensemble to Kaggle

**Afternoon (3 hrs):**
- README polish (target: portfolio-grade, like Fashion/Agentic):
  - Title + tagline
  - Live results badges (Kaggle LB rank, methods covered)
  - Method comparison table
  - WRMSSE results table (all 8+ methods)
  - Architecture diagram of the pipeline
  - Tech stack
  - Repo structure
  - Known limitations
  - "What I learned" — 5 bullet points
- Final clean-up: type hints, docstrings, remove dead code
- Optional: Streamlit demo with model dropdown + forecast visualization

**Deliverable:**
- Final README
- Ensemble result on leaderboard
- Polished GitHub repo

**Commit:** `docs: final README with all method comparisons and learnings`

---

## Repo Structure

```
m5-forecasting-showcase/
├── README.md
├── PLAN.md
├── requirements.txt
├── .gitignore
├── data/                       # gitignored
│   ├── raw/                    # M5 dataset
│   └── processed/              # Engineered features
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_naive_baselines.ipynb
│   ├── 03_classical_methods.ipynb
│   ├── 04_prophet_hierarchical.ipynb
│   ├── 05_feature_engineering.ipynb
│   ├── 06_lightgbm.ipynb
│   ├── 07_deep_learning.ipynb
│   ├── 08_tft.ipynb
│   └── 09_ensemble.ipynb
├── src/
│   ├── data/
│   │   └── loader.py
│   ├── features/
│   │   ├── lags.py
│   │   ├── rolling.py
│   │   ├── calendar.py
│   │   └── pipeline.py
│   ├── models/
│   │   ├── naive.py
│   │   ├── classical.py
│   │   ├── prophet_model.py
│   │   ├── lightgbm_model.py
│   │   ├── lstm_gru.py
│   │   └── tft_model.py
│   ├── evaluation/
│   │   ├── wrmsse.py
│   │   └── leaderboard.py
│   └── utils/
├── reports/
│   ├── 01_eda_findings.md
│   ├── leaderboard.md
│   └── method_comparison.md
├── submissions/                # Kaggle submission CSVs
└── tests/
```

---

## Success Criteria

By Day 10, the project demonstrates:

1. **Method breadth** — 8+ forecasting techniques covered with measured WRMSSE
2. **Production sensibility** — Feature engineering pipeline, evaluation framework, leaderboard tracking
3. **Honest evaluation** — WRMSSE on Kaggle leaderboard, not hand-picked metrics
4. **Documentation** — Each method has a notebook + writeup explaining when it's the right tool
5. **Interview-ready stories** — At least 3 specific debugging/learning moments documented

**Stretch goals:**
- Top 50% on Kaggle public leaderboard
- Hierarchical reconciliation (MinT) implemented
- Deployed Streamlit demo

---

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| ARIMA/SARIMAX takes too long on 30k series | Sample 1,000 series, document trade-off |
| LSTM/TFT runs hit Kaggle 30-hr/week cap | Train smaller models first, scale up only what works |
| Feature engineering memory overflow | Use parquet, float32, categorical dtypes throughout |
| Method choices feel scattered | Group results by family in final README, tell a story |
| TFT setup is complex | `pytorch-forecasting` library handles most of it |

---

## Interview Talking Points (build as you go)

Throughout the 10 days, capture:

1. "When I ran ARIMA on 30k series, I learned [X]. The trade-off is [Y]." (Day 3)
2. "Tweedie loss vs RMSE on M5: [specific result]. Here's why it matters for retail." (Day 7)
3. "LightGBM beat my LSTM by [X%]. The literature says this is consistent for tabular forecasting because..." (Day 8)
4. "TFT's attention weights revealed the model focused on [specific patterns]..." (Day 9)
5. "My ensemble vs the M5 winner's ensemble: differences and lessons." (Day 10)

These are gold for interviews. Write them as you discover them, not at the end.

---

## What This Project Demonstrates to Interviewers

- **Forecasting depth:** From classical statistical methods to modern deep learning
- **Production engineering:** Feature pipeline, evaluation framework, repo organization
- **Honest evaluation:** Public Kaggle leaderboard score, not cherry-picked metrics
- **Trade-off awareness:** When ARIMA wins, when LightGBM wins, when DL is overkill
- **Domain knowledge:** Retail-specific challenges (intermittent demand, hierarchies, holidays)
- **Tool fluency:** statsmodels, prophet, lightgbm, pytorch, pytorch-forecasting, sktime
- **Continuous integration of FedEx experience** (you've used ARIMA/SARIMAX in production — that's a real signal)

---

**Ready to start?** Tell me when to write the Day 1 Claude Code kickoff message.
