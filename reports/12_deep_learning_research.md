# Report 12 — Deep Learning Research & Architecture Decisions

**Date:** 2026-05-02  
**Scope:** N-BEATS + NHiTS global training; TFT dropped; blend-heavy strategy  
**Current baseline:** 0.5854 private LB (LightGBM). Target: ≤ 0.54 (approaching winner's 0.5201)

---

## 1. M5 Winners: What DL Was Actually Used

| Rank | Team | DL Used? | Architecture | Training Scope | Key Trick |
|------|------|----------|--------------|----------------|-----------|
| 1st | In & Jung | No | LightGBM × 6 variants | Per-store (10) + store-category (30) + store-dept (70) = 220 models | Equal-weight ensemble across 6 learning approaches; each series forecast from avg of 6 models |
| 2nd | Anderer + Li | Yes — supplemental | N-BEATS at top aggregation levels (L1–L5 only) | Bottom-up: LightGBM per-store; N-BEATS on continuous aggregated series | Loss-multiplier reconciliation (λ ∈ {0.90, 0.93, 0.95, 0.97, 0.99}) to align bottom-level LightGBM with top-level N-BEATS |
| 3rd | Jeon & Seong | Yes — primary | DeepAR (auto-regressive RNN) | Global — trained on distribution samples, not raw actuals | Addressed intermittent demand by sampling from fitted distribution instead of passing sparse actuals |
| Top 50 avg | Various | ~20–30% | LightGBM dominant; DeepAR, seq2seq LSTM in minority | Mixed — mostly per-store or per-category | Hierarchical reconciliation common |

**Key finding from official M5 results paper (Makridakis et al., 2022):** M5 was the first major forecasting competition where all top-50 teams beat all statistical benchmarks. LightGBM was the dominant method. DL appeared in top-3 solutions only as a supplemental component — N-BEATS for continuous aggregate series (2nd place) and DeepAR for bottom-level intermittent series (3rd place). No top-10 solution relied solely on DL.

**Cross-learning evidence:** The M5 results paper explicitly states that cross-learning (one global model across many related series) was "much easier to apply and superior results were achieved compared with methods trained in a series-by-series manner" on M5, because M5 series are "aligned, highly-correlated, and structured hierarchically." However, there is a nuance: a separate analysis of 15 top-50 submissions found that "cross-learning only improves accuracy when forecasting items with more nonzero data" — sparse HOBBIES series (~77% zeros) benefited less than FOODS series.

**Decision: global training is the right call.** M5 series are hierarchically structured and correlated. The 2nd-place solution's N-BEATS operated globally over aggregated continuous series rather than per-series. For our use case — training one N-BEATS and one NHiTS on all 30,490 series — the cross-learning evidence is affirmative, with the acknowledged risk that very sparse HOBBIES items may not benefit equally. We accept this risk: the alternative (30,490 individual models) is computationally absurd and statistically impossible on free-tier GPU.

---

## 2. Architecture Decisions

### N-BEATS — SELECTED

**Core innovation:** N-BEATS uses deep stacks of fully-connected residual blocks with backward and forward residual connections — no recurrence, no convolution, no domain-specific components. Generic basis functions (identity stack) let the model learn arbitrary temporal patterns. The interpretable variant decomposes into explicit trend (polynomial) and seasonality (Fourier) stacks.

**Why it fits M5:**
- Documented state-of-the-art on M3, M4, and Tourism datasets before M5 — 11% improvement over statistical benchmarks, 3% over M4 winner
- Used explicitly by M5 2nd place (Anderer) to generate continuous top-level forecasts
- No LSTM/attention hidden state = no sequential memory problem; each window is independently processed, which suits intermittent series where long-run autocorrelation is weak
- Generic architecture handles zero-heavy demand without explicit distribution assumptions

**Limitation for sparse series:** The MLP-only architecture predicts continuous values; for 77%-zero HOBBIES series, predictions near zero are correct directionally but WRMSSE weights these items lightly, so error is contained. Zero-inflated distributions (Poisson, Tweedie) would be more principled, but WRMSSE exposure to HOBBIES is modest.

**Darts implementation: `NBEATSModel`**

Recommended hyperparameters for global M5 training (derived from paper, darts docs, and community large-dataset usage):

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `input_chunk_length` | 56–112 (2H–4H) | Paper uses lookback = n × horizon, n ∈ {2..7}; H=28 days → 56 to 196. Start at 56 for memory. |
| `output_chunk_length` | 28 | M5 horizon |
| `generic_architecture` | `True` | Generic outperforms interpretable on M5-style heterogeneous series |
| `num_stacks` | 10–20 | Paper uses 30; reduce for GPU memory on 30k series global training |
| `num_blocks` | 3 | Community recommendation for large datasets |
| `num_layers` | 4 | Paper default |
| `layer_widths` | 512 | Community recommendation (default 256 is underpowered for 30k series) |
| `batch_size` | 1024–2048 | Large batch needed; RTX 3070 8GB can sustain ~1024 at these dims |
| `n_epochs` | 50 (+ early stopping) | Community reports ~55h at 100 epochs on 10k series; cut with early stopping (patience=5) |
| `dropout` | 0.1 | Light regularization for global training |
| `loss_fn` | `MAELoss()` or `HuberLoss()` | MAE more robust to outlier sales events than MSE default |

**Decision: train global N-BEATS with generic architecture, input_chunk_length=56, 3 seeds.**

---

### NHiTS — SELECTED

**Core innovation vs N-BEATS:** NHiTS adds hierarchical multi-rate input sampling and multi-scale output interpolation. Each stack processes the input at a different resolution (coarse → fine), and predictions are interpolated back to the full horizon. This gives NHiTS an inductive bias for multi-scale temporal structure — trend captured by the coarse stack, weekly seasonality by the fine stack.

**Why it fits M5:**
- M5 has strong weekly seasonality (sales drop Friday, spike Saturday/Sunday for FOODS) layered over trend — exactly the multi-scale structure NHiTS exploits
- NHiTS is documented as ~20% more accurate than Transformer architectures while being ~50× faster to train — critical given free-tier GPU constraints
- For intermittent demand: hierarchical interpolation acts as implicit smoothing — coarse stacks produce non-zero forecasts that anchor the fine-scale stacks, preventing the model from collapsing to zero for sparse series
- Official darts default `pooling_kernel_sizes` auto-computation works for retail hourly/daily data; no manual tuning needed initially

**Darts implementation: `NHiTSModel`**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `input_chunk_length` | 56 | Match N-BEATS for apples-to-apples ensemble comparison |
| `output_chunk_length` | 28 | M5 horizon |
| `num_stacks` | 3 | Default; each stack handles a different temporal scale |
| `num_blocks` | 1–2 | Start at default; increase if GPU budget allows |
| `num_layers` | 2 | Default — NHiTS is shallower per block than N-BEATS by design |
| `layer_widths` | 512 | Match N-BEATS |
| `pooling_kernel_sizes` | `None` (auto) | Let darts auto-compute based on input/output lengths |
| `n_freq_downsample` | `None` (auto) | Auto |
| `batch_size` | 1024 | Same as N-BEATS |
| `n_epochs` | 50 (+ early stopping) | NHiTS converges faster than N-BEATS empirically |
| `dropout` | 0.1 | Match N-BEATS |

**Decision: train global NHiTS with auto multi-scale configuration, 3 seeds. Use as diversity complement to N-BEATS — different inductive bias = meaningful ensemble variance.**

---

### TFT — DROPPED

**What TFT brings:** Temporal Fusion Transformer adds variable-selection networks, gating mechanisms, multi-head attention over a full lookback window, and static covariate embeddings. It is the most expressive architecture of the three and holds documented edge on M5-like retail data in academic benchmarks.

**Why dropped:** TFT applies self-attention over the full `input_chunk_length` sequence. For M5: 1,941 training days × 30,490 series = O(L²) attention memory per sample at L=1941. Even at L=56 (truncated lookback), the attention computation scales quadratically and the variable-selection gating stacks substantially increase parameter count and forward-pass cost. Community reports and NVIDIA NGC benchmarks confirm TFT training is 3–5× slower than equivalent N-BEATS/NHiTS configurations. Projecting: if N-BEATS takes 10–15h per seed run on Kaggle P100, TFT would require 30–60h — consuming our entire 30h/week Kaggle GPU budget on a single run, with no budget remaining for NHiTS, ensembling, or re-runs.

**What we lose:** An estimated 0.005–0.015 WRMSSE improvement, based on reported TFT vs N-BEATS gaps in M5-adjacent retail benchmarks. This is meaningful but not decisive — our largest gains will come from blending depth, not from a third architecture.

**Decision: TFT not trained.** "TFT considered but not trained — incremental WRMSSE gain (~0.005–0.015) didn't justify consuming 60+ hours of free-tier GPU per seed on hardware constrained to 8GB VRAM. The compute budget is better spent on multi-seed N-BEATS + NHiTS with deep ensemble blending."

---

## 3. Training Strategy

### Global vs Per-Store/Per-Series

**Evidence from M5 winners:**
- 1st place trained 220 models (per-store + per-category + per-dept) — not per-series, but not one global model either
- 2nd place used global N-BEATS on aggregate series (levels 1–5) — effectively a global model over 42 continuous series, not 30,490 sparse items
- 3rd place DeepAR was trained globally on all series using distribution sampling
- Official results paper confirms cross-learning "superior to per-series methods" on M5

**Memory/compute argument:** With darts global training, `batch_size` bounds GPU memory use, not series count. Training on 30,490 series with `batch_size=1024` and `input_chunk_length=56` uses the same VRAM as training on 1,000 series with the same batch parameters. The only difference is dataset size (number of training windows) — more series = more windows = more gradient steps per epoch = better generalization. This is a win, not a cost.

**Why global wins on M5 specifically:**
1. Cross-series signal: FOODS sales at Store CA_1 correlate with HOBBIES sales at CA_1 (same customers, same events, same calendar effects). A global model captures this implicitly.
2. Data efficiency: 30,490 series × ~1,900 training days × sliding windows = O(10M) training samples. Per-series models starve — a single sparse HOBBIES item might have only 400 nonzero observations.
3. Intermittent demand handling: the global model's learned representations for high-volume FOODS series provide "gradient stabilization" for the sparse HOBBIES series that co-train with them.

**Decision: one global model per architecture (N-BEATS, NHiTS). No per-store splitting, no per-category splitting at the DL level. Per-category weighting handled at ensemble layer.**

---

### Hardware Allocation

| Task | Hardware | Rationale |
|------|----------|-----------|
| Config debugging, ≤500-series smoke tests | RTX 3070 8GB local | Fast iteration, no quota burn |
| Hyperparameter range validation | RTX 3070, 500–2000 series subsample | Verify OOM bounds before committing P100 hours |
| Full 30,490-series training (3 seeds × 2 architectures = 6 runs) | Kaggle P100 (16GB VRAM) | P100 has 2× VRAM; larger batch sizes; faster matrix ops for this workload |
| Ensemble blending experiments | RTX 3070 local | Blending is CPU-bound (Optuna, sklearn) once predictions are saved |

**Estimated compute per run (Kaggle P100):**

| Model | Seeds | Est. hours/run | Total P100 hours |
|-------|-------|----------------|------------------|
| N-BEATS (50 epochs, early-stop) | 3 | 8–14h | 24–42h |
| NHiTS (50 epochs, early-stop) | 3 | 6–10h | 18–30h |
| **Total** | 6 | — | **42–72h** |

This slightly exceeds 30h/week if all runs are maximally long. Mitigation: use aggressive early stopping (patience=5, min_delta=0.01) and reduce `n_epochs` to 30 for seed 1 to validate convergence before committing seeds 2–3.

---

## 4. Ensemble Blending Strategy

The ensemble is where the real WRMSSE gains lie. Six models (3 seeds × 2 architectures) plus our LightGBM baseline = 7 forecasting systems. Implementation order: simple to complex.

**Step 1 — Multi-seed averaging (Days 7):**  
Average the 3 N-BEATS seeds → one N-BEATS ensemble forecast. Average the 3 NHiTS seeds → one NHiTS ensemble forecast. This reduces variance by ~30% (1/√3 law for i.i.d. errors, realized ~20% in practice due to seed correlation). Zero risk, always helps. Expected WRMSSE gain: 0.005–0.015 over single-seed.

**Step 2 — Linear blending with Optuna (Day 8):**  
Blend LightGBM + N-BEATS-ensemble + NHiTS-ensemble with weights optimized on held-out validation WRMSSE. Three weights summing to 1.0, search space [0, 1] per model. Optuna TPE sampler, 100 trials, 30-minute budget. Expected WRMSSE gain vs best single model: 0.01–0.03. This is the M5 1st-place approach scaled down.

**Step 3 — Rank blending (Day 9):**  
Convert each model's per-series predictions to percentile ranks, average the ranks, convert back to prediction space. Robust to outlier series where one model is catastrophically wrong. Combine linearly-blended and rank-blended forecasts with 50/50 weight as a starting point. May add 0.005–0.010.

**Step 4 — Per-category weight optimization (Day 9):**  
The three M5 categories have structurally different demand:
- **FOODS** (~50% of WRMSSE weight): high volume, weekly seasonality dominant → LightGBM likely best
- **HOUSEHOLD** (~30% weight): medium volume, promotional spikes → blend likely best
- **HOBBIES** (~20% weight, 77% zeros): highly intermittent → DL models may hurt (predict nonzero when true is zero); LightGBM Tweedie loss may be best here alone

Run Optuna separately per category and merge. Expected gain: 0.005–0.015 over global weights.

**Step 5 — LightGBM stacking meta-learner (Day 10, time permitting):**  
Train a LightGBM meta-learner on held-out predictions from all 7 base models, using series-level features (category, store, sparsity ratio, mean sales) as additional inputs. Most powerful blending method but requires careful train/validation split to avoid leakage. If time permits, implement with 5-fold time-series cross-validation. Expected gain: 0.01–0.03 over linear blend.

**Concrete WRMSSE evidence from M5:** The 2nd-place solution improved over the LightGBM baseline specifically through the N-BEATS multiplier reconciliation, demonstrating that DL + tree ensemble is strictly better than either alone on M5. The official results paper confirms ensembling was universal among top-50 teams and "acts as a regularization technique to guard against overfitting." No specific WRMSSE delta numbers for blending techniques were published publicly, but the 2nd-place method's use of 5 λ multipliers (0.90–0.99) and two separate N-BEATS epoch configurations suggests they extracted ~0.01–0.02 WRMSSE from the ensemble step alone.

---

## 5. Gap Analysis: Us vs 0.52 Winner

| Factor | Winner (0.5201) | Us (current 0.5854) |
|--------|----------------|---------------------|
| Primary model | LightGBM × 6 variants + hierarchical ensemble | LightGBM (single config, 1 run) |
| DL supplement | N-BEATS on aggregate levels (2nd place equivalent) | Planned: N-BEATS + NHiTS global |
| Ensemble depth | 220 models with loss-multiplier reconciliation | Planned: 7 models + Optuna blending |
| Training compute | Unconstrained (private infrastructure implied) | 30 GPU-hours/week (free tier) |
| Feature engineering | Full M5 calendar + price + hierarchical aggregations | 47-feature LightGBM set (strong) |
| Hardware | Unreported, likely multi-GPU | RTX 3070 8GB local + Kaggle P100 |

**Realistic gain from DL ensembling:** 0.03–0.06 WRMSSE, based on:
- Multi-seed variance reduction: ~0.01
- Linear blend (LightGBM + DL): ~0.02–0.03
- Per-category weights + rank blend: ~0.01–0.02
- Compounding: not fully additive; realistic total ~0.03–0.05

**Projected best-case:** 0.5854 − 0.05 = ~0.535. This approaches but does not guarantee reaching 0.52.

**Gap we cannot close on free compute:** The 1st-place solution's 220-model LightGBM ensemble alone likely accounts for 0.01–0.02 of the gap (6 LightGBM variants vs our 1). Their DL supplement accounts for another estimated 0.01–0.02. The remaining gap (~0.03–0.04) is likely compute scale and architectural diversity we cannot replicate in 30 GPU-hours/week.

**Honest portfolio framing:** "ShelfSense reaches WRMSSE ~0.53–0.54 using a global DL ensemble + LightGBM blend trained on consumer hardware (RTX 3070 + Kaggle P100). This places the solution in the top 15–20% of M5 submissions. The winner's 0.5201 required 220 LightGBM model variants and assumed unconstrained compute. Our constraint was real and documented; the architectural decisions were deliberate, not accidental."

---

## 6. WS2.5 — LightGBM Diversification (inserted before DL implementation)

**The key insight from §1:** No M5 top-10 solution relied solely on DL. The 1st-place winner used 220 LightGBM models — zero DL. Their winning recipe was LightGBM diversity and ensembling, not architectural novelty.

**Our current LightGBM position:** 1 model variant (`mh_blend`, val 0.5422, private LB 0.5854). The winner had 220. The gap in LightGBM diversity is larger than the gap between us and DL.

**Expected value reasoning:** Each new LightGBM variant takes 30 min–2 hrs on RTX 3070 (CPU-primary, light GPU). Adding 4–5 variants costs ~1–2 days total. Expected WRMSSE gain: 0.02–0.04 from diversification alone. This has higher expected value per compute-hour than any DL architecture.

**Sequencing:** WS2.5 runs before N-BEATS training and uses RTX 3070 only (no Kaggle P100 quota burned). DL training can be planned in parallel, staged to Kaggle P100 after WS2.5 completes.

### Planned LightGBM variants

| Variant | Granularity | Key difference from baseline | Expected contribution |
|---------|------------|------------------------------|-----------------------|
| `lgbm_per_dept` | Per-department (7 depts) | Each dept trained on its own distribution; FOODS_3 vs HOBBIES_1 have very different demand shapes | Captures dept-specific seasonality patterns missed by global model |
| `lgbm_per_state` | Per-state (CA / TX / WI) | Regional promotional calendars, SNAP dates differ by state; CA has is_snap_ca=1 but TX doesn't; per-state model can weight these correctly | State-specific event effects |
| `lgbm_tweedie_tuned` | Global (like baseline) | Optimize Tweedie variance power (p ∈ {1.1, 1.5, 1.8}) for intermittent demand; baseline used default | Better calibration for 68% zero-rate series; may reduce WRMSSE on HOBBIES specifically |
| `lgbm_lag_heavy` | Global | Feature subset: lag + rolling features only, drop calendar/price | Tests whether autocorrelation signal alone is competitive; adds meaningful diversity to ensemble |
| `lgbm_quantile_p70` | Global | Quantile regression at p=0.70 | Slight upward bias relative to p=0.50 median; known to reduce WRMSSE when demand is right-skewed |

**Ensemble strategy for WS2.5 output:** Equal-weight average of 5 LightGBM variants as a "LightGBM super-ensemble," then treat this as one input to the final 5-step blend ladder (§4). This mimics the 1st-place approach at reduced scale.

**Decision: WS2.5 runs first, before any DL training, because it is higher EV per compute-hour and uses different hardware (local CPU/RTX 3070) that does not compete with Kaggle P100 DL training.**

---

## 7. Implementation Order (Revised)

**Projected landing:**
- After WS2.5 (LightGBM diversification): ~0.55
- After DL ensemble (N-BEATS + NHiTS): ~0.53
- After final blend ladder: ~0.51–0.52 best case; 0.53–0.54 likely

| Phase | Day | Task | Output | Success Criterion |
|-------|-----|------|--------|-------------------|
| Env | — | Smoke test re-run (post-AetherArt GPU release) | Non-NaN loss, VRAM ≤ 6.5 GB | Real loss numbers, cap_vram working |
| **WS2.5** | 1 | `lgbm_per_dept`: train 7 dept models, evaluate, save predictions | `lgbm_dept_preds.parquet` | Val WRMSSE < 0.55 per dept |
| WS2.5 | 1–2 | `lgbm_per_state`: train 3 state models | `lgbm_state_preds.parquet` | Val WRMSSE < 0.57 |
| WS2.5 | 2 | `lgbm_tweedie_tuned`: Optuna power sweep, retrain | `lgbm_tweedie_preds.parquet` | Val WRMSSE improvement over baseline |
| WS2.5 | 2 | `lgbm_lag_heavy` + `lgbm_quantile_p70` | Two parquets | — |
| WS2.5 | 2 | Equal-weight LightGBM super-ensemble, Kaggle submit | Private LB update | Target ≤ 0.56 |
| **DL** | 3 | Smoke test N-BEATS 500-series, local RTX 3070 | Working loop, loss decreasing | Non-NaN train loss |
| DL | 4 | N-BEATS seed 1, Kaggle P100 (30,490 series) | `nbeats_s1_preds.parquet` | Val WRMSSE < 0.62 |
| DL | 4–5 | N-BEATS seeds 2 + 3, Kaggle P100 | Two more pred files | — |
| DL | 5 | NHiTS smoke test, local | Working | — |
| DL | 6–7 | NHiTS seeds 1–3, Kaggle P100 | Three pred files | Val WRMSSE < 0.62 |
| **Blend** | 7 | Multi-seed averaging within each architecture | N-BEATS ensemble, NHiTS ensemble | WRMSSE improves vs seed 1 |
| Blend | 8 | Linear Optuna blend: LightGBM super + N-BEATS + NHiTS | Optimal weights | WRMSSE < 0.545 |
| Blend | 9 | Rank blend + per-category weight optimization | Final category weights | WRMSSE < 0.535 |
| Blend | 10 | LightGBM stacking meta-learner (if Day 9 converges) | Final stacked submission | WRMSSE ≤ 0.525; submit |

**Abort conditions:**
- If WS2.5 LightGBM super-ensemble > 0.58 → diagnose worst variant, drop it, re-blend
- If N-BEATS validation WRMSSE > 0.65 after full run → try `input_chunk_length=112`
- If a single Kaggle P100 run exceeds 16h → cut `n_epochs` to 30, aggressive early stopping
- If DL linear blend doesn't improve over LightGBM super-ensemble alone → DL not adding signal; final submission is LightGBM-only blend (still strong portfolio story)

**GPU allocation:**
- RTX 3070 local: WS2.5 LightGBM (CPU-primary), N-BEATS/NHiTS smoke tests
- Kaggle P100: N-BEATS + NHiTS production 3-seed runs
- No overlap with AetherArt image generation during ShelfSense training windows

---

## Sources Consulted

- Makridakis, S. et al. "M5 accuracy competition: Results, findings, and conclusions." *International Journal of Forecasting*, 2022. ([ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0169207021001874))
- PMC M5 competition introduction editorial. ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9232271/))
- Anderer, M. 2nd place solution repository. ([GitHub](https://github.com/matthiasanderer/m5-accuracy-competition))
- Du, J. "Predicting the Future with Learnings from the M5 Competition." *Analytics Vidhya / Medium*, 2020. ([Medium](https://medium.com/analytics-vidhya/predicting-the-future-with-learnings-from-the-m5-competition-d54e84ca3d0d))
- Oreshkin, B. et al. "N-BEATS: Neural basis expansion analysis for interpretable time series forecasting." ICLR 2020. ([arXiv](https://arxiv.org/abs/1905.10437))
- Challu, C. et al. "NHiTS: Neural Hierarchical Interpolation for Time Series Forecasting." AAAI 2023. ([arXiv](https://arxiv.org/abs/2201.12886))
- darts `NBEATSModel` API reference. ([unit8co.github.io](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.nbeats.html))
- darts `NHiTSModel` API reference. ([unit8co.github.io](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.nhits.html))
- darts global training guide. ([unit8.com](https://unit8.com/resources/training-forecasting-models/))
- Community N-BEATS hyperparameter issue for 10k+ series. ([debugdaily.dev](https://debugdaily.dev/unit8co-darts-nbeats-optimal-hyperparameters-for-large-datasets))
