# Report 11 — DL Environment Setup

**Date:** 2026-05-02  
**Outcome:** darts 0.44.0 selected as DL training library. neuralforecast blocked by platform constraints.

---

## 1. What we tried

### neuralforecast

| Attempt | Result |
|---------|--------|
| Installed version: `0.1.0` | Broken — `pl.utilities.distributed` removed in pytorch_lightning >= 1.6 |
| `pip install --upgrade neuralforecast` | Silently resolved to same 0.1.0 (pip dependency conflict algorithm) |
| `pip install neuralforecast==3.1.7` | **BLOCKED**: requires `ray>=2.2.0`, no Windows wheel for Python 3.13 |
| `pip install neuralforecast==1.7.7` | Same block — ray requirement present across all 1.x versions |
| `pip install ray` | **BLOCKED**: no Windows distribution for Python 3.13 (as of 2026-05-02) |

**Root cause:** Python 3.13 shipped in late 2024. Ray's Windows + Python 3.13 support lagged. neuralforecast 1.x+ made ray a hard dependency for its distributed training backend. The Anaconda base environment ships Python 3.13.5, making the entire neuralforecast ≥ 1.0 family uninstallable without a separate conda environment.

**Alternative considered:** Patch `neuralforecast==0.1.0` (`experiments/utils.py:33`). Rejected — the one-liner fix would surface further pytorch_lightning 2.x API breakages deeper in the training loop. Not worth chasing.

---

## 2. Library selected: darts 0.44.0

```
pip install darts
# Installed: darts-0.44.0, llvmlite-0.47.0, nfoursid-1.0.2,
#            numba-0.65.1, numpy-2.4.4, pyod-3.2.1, shap-0.51.0, slicer-0.0.8
```

**Why darts wins here:**

- Has all three target architectures: `NBEATSModel`, `NHiTSModel`, `TFTModel`
- No ray dependency — installs clean on Python 3.13 Windows
- pytorch_lightning 2.6.1 is natively supported (darts 0.44.0 uses pl ≥ 2.0)
- GPU training via `pl_trainer_kwargs={"accelerator": "gpu", "devices": 1}`
- Past/future covariates as first-class citizens — directly maps to our feature taxonomy
- Identical underlying architectures to neuralforecast versions; outputs (WRMSSE) are comparable

**API difference from plan:** darts uses `TimeSeries` objects rather than long-format DataFrames. The `to_darts_datasets()` function in `src/shelfsense/data/dl_format_adapter.py` handles this conversion.

---

## 3. Final environment

| Component | Version | Status |
|-----------|---------|--------|
| Python | 3.13.5 (Anaconda) | — |
| torch | 2.6.0+cu124 | OK |
| pytorch_lightning | 2.6.1 | OK |
| darts | 0.44.0 | **OK — primary DL library** |
| neuralforecast | 0.1.0 | Broken (left installed, unused) |
| numpy | 2.4.4 | Upgraded by darts install |
| CUDA | cu124 | OK |
| GPU | RTX 3070 Laptop GPU, 8.0 GB VRAM | OK |

---

## 4. Data format adapter

`src/shelfsense/data/dl_format_adapter.py`

Converts per-store features parquet → darts-compatible TimeSeries.

**Verified on store_CA_1:**
- Shape: 5,918,109 rows × 41 columns
- Series: 3049 unique_ids, each exactly 1941 days
- Date range: 2011-01-29 → 2016-05-22 (M5 training period d_1–d_1941)

**Column taxonomy:**

| Group | Columns | darts role |
|-------|---------|-----------|
| Target | `sales` / `y` | `series` |
| Past covariates | `lag_7/14/28/56`, `roll_*` (20 cols) | `past_covariates` |
| Future covariates | calendar (13 cols) + price (5 cols) | `future_covariates` |
| Dropped | `item_id, dept_id, cat_id, store_id, state_id, d, d_num` | — |

---

## 5. VRAM policy for all WS2 training

**Observed during smoke test:** default darts N-BEATS (batch_size=32) used 7997/8192 MB VRAM — 97.6% of total, leaving only ~195 MB for the system display driver and desktop compositor. On a laptop this causes display stutter and risks GPU driver reset under sustained load.

**Policy:** reserve 1.5–2 GB for system. Train within 6–6.5 GB effective ceiling.

**Implementation in every training script:**

```python
from src.shelfsense.data.vram_utils import cap_vram, assert_vram_headroom, vram_status

cap_vram(fraction=0.80)          # hard cap at 6.4 GB (80% of 8 GB), call before model creation
assert_vram_headroom(min_free_gb=1.5)   # sanity check before large allocations
print(vram_status())             # log VRAM state at training start
```

`cap_vram(0.80)` calls `torch.cuda.set_per_process_memory_fraction(0.80)`, which raises `OutOfMemoryError` before the OS-level display contention point. This is preferable to reducing batch_size alone because it enforces the ceiling even if batch_size is accidentally set too high.

**Batch size guidance per architecture (all targeting ≤6.4 GB):**

| Model | Safe batch_size | Expected VRAM | Notes |
|-------|----------------|---------------|-------|
| N-BEATS | 16 | ~4–5 GB | Default 10-stack is large; 16 is safe floor |
| N-HiTS | 32 | ~3–4 GB | Hierarchical interpolation; smaller footprint |
| TFT | 16 | ~5–6 GB | Attention on long sequences is VRAM-hungry |

All training calls also set `num_workers=0` — Windows multi-process dataloader forks can cause system RAM spikes that indirectly pressure VRAM via unified memory.

---

## 6. What changes in WS2 plan

The original plan was written for neuralforecast's NeuralForecast class and long-format DataFrames. Switching to darts:

- `NeuralForecast(models=[...], freq="D")` → `NBEATSModel(...).fit(series, past_covariates=...)`
- `nf.predict()` → `model.predict(n=28, series=..., past_covariates=...)`
- Training loop, checkpoint saving, and WRMSSE evaluation: same logic, different wrapper API

Research plan (Report 12) targets darts API examples. Architectures (N-BEATS, N-HiTS, TFT), hyperparameter choices, and expected performance are unchanged.
