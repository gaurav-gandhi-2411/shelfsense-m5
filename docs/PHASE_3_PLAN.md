# Phase 3 — Production-Grade Refactor Plan

## Goal

Transform the ShelfSense-M5 repo from a research-grade flat-script pipeline into a portfolio-grade engineered system. The result demonstrates senior-level engineering judgment to a hiring manager evaluating the repo cold: clean structure, reproducible from a fresh clone in under two hours, properly tested, properly orchestrated, fully documented.

We are not building a live forecasting service. There is no API, no scheduler, no monitoring. The artifact is a batch ML training pipeline executed end-to-end — production-grade in the sense of being reproducible, observable, testable, and maintainable, not in the sense of serving requests.

## Non-goals

- No live API or web service.
- No deployment to cloud beyond what's needed for CI (GitHub Actions) and notebook reproducibility (Kaggle).
- No replacement of the model approach. The Phase 2 result (private LB 0.5693, or whatever the yearly-lag experiment produces) is locked. Phase 3 is structural.
- No additional model experiments after Phase 3 starts. New model variants become trivially easy to add post-refactor; we just don't run them as part of this work.

## Target end state

A new contributor can:

```
git clone https://github.com/gaurav-gandhi-2411/shelfsense-m5
cd shelfsense-m5
make setup        # builds Docker image with pinned deps
make data         # downloads M5 data, builds features, validates schemas
make train        # runs full Dagster pipeline → models → ensemble
make submit       # submits best model to Kaggle
make test         # full test suite
make report       # regenerates charts and leaderboard
```

Total wall time on a fresh clone with a free Kaggle GPU notebook or comparable Linux box: under two hours.

The repo's GitHub front page tells a complete engineering story without needing the visitor to clone or run anything.

## Stages

Phase 3 breaks into six stages. Estimated 13–19 working days of focused work, but no calendar pressure.

### Stage 1 — Repo restructure and CLI

Replace flat `scripts/05_*, 06_*, ...` with a proper Python package. Project layout:

```
shelfsense-m5/
├── pyproject.toml              # uv-managed deps, pinned via uv.lock
├── uv.lock
├── Dockerfile
├── docker-compose.yml          # services: train, mlflow, dagster
├── Makefile                    # one-liner commands above
├── shelfsense/                 # the package
│   ├── __init__.py
│   ├── cli.py                  # Typer entry point
│   ├── config/                 # Hydra configs (YAML)
│   │   ├── config.yaml
│   │   ├── data/
│   │   ├── features/
│   │   ├── model/
│   │   └── ensemble/
│   ├── data/
│   │   ├── load.py             # M5 CSV loaders
│   │   ├── validate.py         # Pandera schemas
│   │   └── splits.py           # train/val/eval boundaries
│   ├── features/
│   │   ├── lags.py
│   │   ├── rolling.py
│   │   ├── calendar.py
│   │   ├── price.py
│   │   ├── hierarchy.py
│   │   ├── pipeline.py         # orchestrates feature build
│   │   └── registry.py         # named feature sets
│   ├── models/
│   │   ├── base.py             # abstract Model interface
│   │   ├── lightgbm/
│   │   │   ├── tweedie.py
│   │   │   ├── multihorizon.py
│   │   │   ├── store_dept.py
│   │   │   └── ensemble.py
│   │   └── classical/          # ETS/ARIMA/Prophet wrappers
│   ├── evaluation/
│   │   ├── wrmsse.py           # exact Kaggle-matching evaluator
│   │   └── reports.py          # leaderboard generator
│   ├── orchestration/
│   │   └── assets.py           # Dagster assets
│   ├── tracking/
│   │   └── mlflow_utils.py     # log_model, log_metrics, register
│   └── visualization/
│       └── charts.py           # ChartCanvas
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/               # 100-series synthetic data
├── notebooks/
│   ├── 01_eda.ipynb            # polished
│   └── 02_failure_analysis.ipynb  # new — store×dept val→private divergence
├── data/                       # DVC-tracked
├── reports/                    # markdown findings
├── docs/
│   ├── ARCHITECTURE.md
│   ├── MODELS.md
│   ├── RUNBOOK.md
│   └── CONTRIBUTING.md
├── .github/
│   └── workflows/
│       ├── ci.yml              # lint, type-check, test
│       └── release.yml         # tag → Docker image to GHCR
└── README.md                   # rewritten with final scores
```

CLI examples via Typer:

```
shelfsense data download
shelfsense features build --config-name features/default
shelfsense train tweedie-mh --tvp 1.3 --seed 42
shelfsense train store-dept --slices all
shelfsense ensemble --candidates tvp_13,store_dept --method optuna
shelfsense submit --variant best --kaggle
```

Hydra makes every experiment a YAML config, fully parameterized. Multi-run sweeps via `--multirun`. Example:

```
shelfsense train tweedie-mh --multirun model.tvp=1.3,1.5,1.7
```

This becomes the new way to run any experiment. The numbered scripts move to `scripts/legacy/` with a README pointing to the CLI equivalents, kept for reference but not maintained.

**Deliverables**: package skeleton, all existing logic ported to package, Typer CLI, Hydra configs for all current experiments, `scripts/legacy/` archive.

**Effort**: 3–4 days.

### Stage 2 — Environment, dependencies, Docker

Replace the unpinned `requirements.txt` with `pyproject.toml` managed by `uv` (faster than pip, lock file is deterministic, supports extras cleanly).

Two extras:
- `[dev]` — pytest, ruff, black, mypy, pre-commit
- `[dl]` — darts, torch, pytorch-lightning (optional install for the future N-BEATS path)

Dockerfile:
- Base: `nvidia/cuda:12.4-runtime-ubuntu22.04` (works without GPU; needed if N-BEATS path is ever taken)
- Multi-stage build: builder installs deps, runtime is slim
- Non-root user
- Volume mounts for `data/`, `mlruns/`, `dagster_home/`

Compose services:
- `train` — runs the pipeline
- `mlflow` — UI on :5000
- `dagster` — UI on :3000

`make setup` builds all images and primes MLflow + Dagster local stores.

**Deliverables**: `pyproject.toml`, `uv.lock`, `Dockerfile`, `docker-compose.yml`, `Makefile`. `make setup` works on a fresh Linux box.

**Effort**: 1–2 days.

### Stage 3 — Data versioning and validation

DVC tracking for `data/raw/`, `data/processed/`, `data/models/`. Remote: Google Drive (free) — not for collaboration, just as an offsite snapshot. Local-only is also acceptable; the goal is `dvc pull` reproducibility, not multi-user sync.

Pandera schemas (lighter than Great Expectations) for:
- Raw M5 CSVs (column types, value ranges, foreign-key consistency between calendar/sales/prices)
- Feature parquets (column count, dtype per column, NaN rules per feature group)
- Predictions parquets
- Submission CSVs (Kaggle format compliance)

Schema mismatches fail loudly via `shelfsense data validate`, which is mandatory in CI.

This formalizes the gap exposed by the Stage A Phase 2 incident: silently changing column counts in the parquets caused F19.5-class confusion. Pandera plus DVC means schema drift is caught at the boundary, and data versions are pinned per experiment.

**Deliverables**: `dvc.yaml`, `.dvc/config`, `shelfsense/data/validate.py` with all schemas, CI step running validation.

**Effort**: 2–3 days.

### Stage 4 — Orchestration and tracking

Dagster as the orchestrator. Each pipeline output is a versioned asset:

```
raw_sales_csv  ─┐
raw_calendar   ─┼→  raw_validated  →  features  →  per_model_predictions  →  ensemble  →  submission
raw_prices    ─┘
```

Each asset declares its dependencies, materialization function (the actual code), and metadata (Hydra config used, schema version). Dagster's asset graph view is the system documentation — a hiring manager opening `dagster dev` sees the whole pipeline in one screen.

Each model variant (tvp_13, tvp_17, rmse_mh, store_dept, ylag_mh, ...) is a parameterized asset. Adding a new variant is a 20-line file in `shelfsense/models/lightgbm/`, registered as an asset in `orchestration/assets.py`. No script copying.

MLflow tracks every materialization:
- Params: full Hydra config
- Metrics: per-slice and overall same-origin val WRMSSE, public LB, private LB
- Artifacts: model pickle, predictions parquet, submission CSV, feature importance plot
- Tags: git commit hash, data version (DVC), feature set version

Local backend: SQLite + filesystem artifact store. No cloud dependency. `make mlflow-ui` opens the comparison view.

`shelfsense experiments compare run1 run2` is a CLI shortcut to MLflow's run comparison, useful for the README narrative.

**Deliverables**: `shelfsense/orchestration/assets.py`, `shelfsense/tracking/mlflow_utils.py`, full pipeline runnable via `dagster materialize`, all existing model variants ported as assets.

**Effort**: 3–4 days.

### Stage 5 — Testing and CI

Unit tests:
- WRMSSE evaluator (existing 24 tests preserved + extended for edge cases)
- Each feature transformation (lags shift correctly, rolling windows are NaN-free past warm-up, no future leakage)
- Each model wrapper's `fit` and `predict` interface
- Hydra config loading and overrides
- Pandera schema enforcement (negative tests: bad data must fail)

Integration tests:
- Full pipeline on a 100-series synthetic fixture in under five minutes
- Asserts: no NaN in outputs, schema match at every boundary, expected WRMSSE range
- Runs in CI on every push

Pre-commit hooks:
- ruff (linting + import sorting)
- black (formatting)
- mypy (type checking, lenient first pass)

GitHub Actions:
- `ci.yml` on push/PR: install deps, lint, type-check, run unit + integration tests
- `release.yml` on tag: build Docker image, push to GHCR

Test coverage target: 70% on `shelfsense/` package (not the legacy scripts). Coverage reported in CI.

**Deliverables**: `tests/` with all test files, `.pre-commit-config.yaml`, `.github/workflows/ci.yml` and `release.yml`, coverage badge in README.

**Effort**: 2–3 days.

### Stage 6 — Documentation polish

The README is the most important file in the repo. It is the hiring manager's first read.

README structure (rewritten from the existing version):
1. Hero: best private LB, baseline, reduction, one-line architecture summary, single best chart
2. TL;DR: 3–4 paragraphs covering the dataset, the journey, the result, the engineering takeaway
3. Architecture: link to ARCHITECTURE.md, embed the pipeline diagram
4. The Journey: condensed version of existing prose, updated with WS2.5 results, retains the three engineering stories (HOBBIES/Tweedie, top-down hierarchy, ensemble inversion) plus a fourth on val→private divergence (the lessons from RMSE-MH and store×dept)
5. Engineering decisions made: existing table, extended
6. Final results: full leaderboard table
7. What I'd do next: existing list, updated with realistic expected gains
8. Reproduce: `make` commands, with timing per command
9. Project structure: link to docs/

`docs/ARCHITECTURE.md`:
- System diagram (mermaid)
- Pipeline stage descriptions (raw → features → models → ensemble → submission)
- Data flow with DVC asset versions
- Why these tools (Hydra, Dagster, MLflow, DVC, Pandera) — short rationale per choice

`docs/MODELS.md`:
- Per-model rationale with hyperparameters, training data slice, val and private LB scores in a single table
- Feature importance plots per model
- The val→private divergence section: a calibration discussion explaining why we trust private over val, with the data backing it

`docs/RUNBOOK.md`:
- "How to add a new model variant" with the actual command sequence
- "How to debug a failed CI run"
- "How to update DVC-tracked data"
- "How to reproduce a specific past experiment from MLflow"

`docs/CONTRIBUTING.md`:
- Even though there will be no contributors, this signals "this person knows how to run an open-source project"
- Pre-commit setup, test running, PR conventions, commit message style

Notebook polish:
- `notebooks/01_eda.ipynb` — clean, narrative comments, key chart at top
- `notebooks/02_failure_analysis.ipynb` — new, short, focused on val→private divergence with the data from RMSE-MH and store×dept. This doubles as portfolio content (honest analysis is more impressive than glossy success stories)

**Deliverables**: rewritten README, four `docs/` files, two polished notebooks, regenerated portfolio charts with final numbers.

**Effort**: 2–3 days.

## Cross-cutting concerns

These appear in multiple stages but worth calling out:

**Hydra + MLflow integration.** Hydra is the source of truth for params. MLflow logs the resolved config per run. A run can be reproduced exactly by checking out the git commit, restoring the DVC data version, and re-running with the same Hydra config.

**Cache invalidation.** The parameter-fingerprint hash pattern from Phase 2 (Claude Code's store×dept work) becomes the standard. Every model artifact's filename includes a hash of (params, feature set version, data version). This makes F19.5-class silent cache misses structurally impossible.

**Determinism.** Every random seed is in Hydra config and logged to MLflow. Re-running with the same config and data produces bit-identical results. This is testable: an integration test verifies determinism on the synthetic fixture.

**Portability.** The pipeline runs on any Linux box with Docker + NVIDIA driver. Tested on: Ubuntu 24 (developer machine), Kaggle Notebooks (free GPU), GitHub Actions runner (CPU only, for CI integration tests).

## Risk and mitigation

| Risk | Mitigation |
|------|-----------|
| Dagster learning curve eats Stage 4 time | Budget reflects this. Fallback: simpler Prefect or pure CLI orchestration if Dagster turns into a 2-week sink. |
| DVC remote on Google Drive flaky | Local-only DVC is also acceptable. The portability goal is `dvc pull` reproducibility, which works either way. |
| Hydra + Typer integration awkward | Hydra has first-class Typer integration via `@hydra.main` decorator; if it doesn't work cleanly, fall back to Hydra-only and skip Typer. |
| Test coverage below target | Acceptable to ship at 50% if tests cover the load-bearing parts (evaluator, feature pipeline, ensemble). Coverage is a signal, not a goal. |
| README rewrite drifts into rewrite-the-whole-thing | Hard time-box: 1.5 days for README, 1.5 days for the four docs/ files. |

## Out of scope explicitly

- Live serving (no FastAPI, no model server)
- Multi-user collaboration tooling (single-developer project)
- Cloud deployment (no Kubernetes, no ECS, no Terraform)
- Streaming or real-time inference
- A/B testing infrastructure
- Drift detection for live predictions (no live predictions)
- Kaggle's six-notebook write-up plan (replaced by the polished `01_eda` and new `02_failure_analysis`)

These are real production concerns but they are not appropriate for a Kaggle competition portfolio piece. Including them would feel performative; excluding them with a clear rationale signals judgment.

## Success criteria

A senior data scientist or hiring manager opening this repo for the first time should be able to, in 15 minutes:

1. Understand what the project does and what the result is (README hero + TL;DR)
2. See that the engineering is real (file structure, Dockerfile, CI badge, test coverage badge, MLflow comparison screenshot)
3. Understand a key technical decision and why it was made (the val→private divergence story is the strongest signal of engineering judgment in the repo)
4. Believe they could check out the repo and run it end-to-end on their own machine

If those four things land, Phase 3 was worth doing.

## Sequencing within each stage

Each stage above is broken into commits, not into "go away for three days and reappear with everything done." Suggested commit cadence:

- Stage 1: 8–12 commits, each ~150 lines net
- Stage 2: 4–6 commits
- Stage 3: 5–8 commits
- Stage 4: 6–10 commits
- Stage 5: 5–8 commits
- Stage 6: 4–6 commits

Total: ~32–50 commits. Each one is a reviewable unit. The git log itself becomes part of the portfolio signal.
