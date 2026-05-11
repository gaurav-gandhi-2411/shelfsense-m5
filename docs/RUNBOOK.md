# ShelfSense M5 — Operations Runbook

## DVC Remote Setup (Google Drive)

DVC is configured with a Google Drive remote named `gdrive`. Authentication uses a GCP service
account so `dvc push/pull` works non-interactively in CI and on fresh clones.

### One-time setup

**1. Create a GCP service account**

```bash
PROJECT_ID=your-gcp-project
gcloud iam service-accounts create shelfsense-dvc \
    --display-name "ShelfSense DVC" \
    --project "$PROJECT_ID"
```

**2. Download the JSON key**

```bash
gcloud iam service-accounts keys create secrets/gdrive-sa.json \
    --iam-account shelfsense-dvc@${PROJECT_ID}.iam.gserviceaccount.com
```

`secrets/` is gitignored — the key never leaves the local machine.

**3. Share the Google Drive folder with the service account**

Open the Drive folder whose ID is in `.dvc/config` (the `gdrive://` URL) and share it with the
service account email `shelfsense-dvc@<PROJECT_ID>.iam.gserviceaccount.com` — Editor access.

**4. Configure the local DVC credential path**

```bash
dvc remote modify --local gdrive gdrive_use_service_account true
dvc remote modify --local gdrive gdrive_service_account_json_file_path secrets/gdrive-sa.json
```

These write to `.dvc/config.local`, which is gitignored by DVC.

**5. Verify connectivity and pull data**

```bash
dvc pull          # downloads data/raw/m5-forecasting-accuracy/ and data/processed/features/
```

---

## How to add a new model variant

Adding a variant follows six steps. The existing `model_ylags` asset is a good reference — it adds
a feature set extension on top of `model_tvp_13` with minimal new code.

**Step 1 — Create the Hydra config**

```bash
# shelfsense/config/model/<variant>.yaml
# Copy the nearest existing config and change only the parameters that differ.
cp shelfsense/config/model/tweedie_mh_tvp13.yaml shelfsense/config/model/<variant>.yaml
# Edit the new file — do not duplicate unchanged keys.
```

**Step 2 — Add the `@asset` in `shelfsense/orchestration/assets.py`**

```python
_VARIANT_CFG = OmegaConf.load("shelfsense/config/model/<variant>.yaml")

@asset(
    config_schema={**_MODEL_CONFIG_SCHEMA, "model_dir": Field(str, default_value="data/models/<variant>")},
    description="One-line description of what this variant tests.",
)
def model_<variant>(context, features_validated: str, mlflow_resource: MLflowResource) -> dict:
    from shelfsense.models.lightgbm.multihorizon import DEFAULT_FEATURE_COLS, MultiHorizonTrainer
    cfg = context.op_config
    trainer = MultiHorizonTrainer(_VARIANT_CFG)
    result = trainer.fit(
        features_dir=features_validated,
        model_dir=cfg["model_dir"],
        feature_cols=DEFAULT_FEATURE_COLS,
        raw_dir=cfg["raw_dir"],
        num_boost_round_override=10 if cfg["test_mode"] else None,
        horizon_override=1 if cfg["test_mode"] else None,
    )
    # log to MLflow ...
    return {"model_dir": result["model_dir"], "val_wrmsse": result["val_wrmsse"]}
```

**Step 3 — Add asset checks**

At minimum add the pkl-count check and the val-WRMSSE range check:

```python
@asset_check(asset="model_<variant>", description="model_<variant> must write exactly 28 h_*.pkl files.")
def check_model_<variant>_pkl_count(model_<variant>: dict) -> AssetCheckResult:
    return _check_mh_pkl_count(model_<variant>["model_dir"])

@asset_check(asset="model_<variant>", description="model_<variant> val_wrmsse must be in (0.5, 1.5).")
def check_model_<variant>_val_wrmsse(model_<variant>: dict) -> AssetCheckResult:
    return _check_val_wrmsse(model_<variant>["val_wrmsse"])
```

**Step 4 — Register in the CLI**

Add the asset name to `_ASSET_NAMES` in `shelfsense/cli.py` (the list is topologically ordered —
insert after `model_ylags`). The `shelfsense materialize --asset model_<variant>` command becomes
available immediately.

**Step 5 — Add a unit test in `tests/unit/test_assets.py`**

The minimal test is the config-loading check that the existing `model_tvp_13` test uses. Add a
`test_<variant>_cfg_loads()` that instantiates the config and checks the `objective` key.

**Step 6 — Run the full check suite**

```bash
uv run pytest tests/unit/ -x        # unit tests
pre-commit run --all-files           # lint, format, type check
uv run --no-dev python -c "from shelfsense.orchestration.assets import model_<variant>"
```

The last command verifies the asset is importable without dev extras — catches missing `__all__`
exports or accidental dev-only imports.

---

## How to debug a failed CI run

**1. Reproduce the failure locally with the same command CI ran**

The CI command is the canonical source of truth. Find it in `.github/workflows/ci.yml`:

```bash
# Install exactly as CI does
uv sync --frozen --extra dev

# Run the same test command
uv run pytest tests/ --cov=shelfsense --cov-report=term-missing --cov-fail-under=60 -p no:forked

# Run the same lint commands
uv run ruff check shelfsense/ tests/
uv run mypy shelfsense/ --ignore-missing-imports --no-strict-optional --allow-untyped-defs
```

Use `uv sync --frozen` (not `uv sync`) to reproduce the exact pinned dependency set from
`uv.lock`. Floating installs may mask or introduce version-specific failures.

**2. Common failure modes**

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `ModuleNotFoundError` on CI but not locally | New runtime dep placed in `[dev]` instead of `[project.dependencies]` | Move dep; run `uv run --no-dev python -c "import <module>"` to verify |
| `SchemaError` in `data validate` | Pandera schema change or data schema drift | Update the schema in `shelfsense/data/schemas.py`; add a negative test case |
| Coverage drop below 60% | New code path not covered | Add unit test; check `--cov-report=term-missing` output for uncovered lines |
| `mypy` error on new code | Missing type annotation or wrong return type | Add annotations; `--allow-untyped-defs` covers function bodies but not return types |
| Pre-commit hook failure | Lint error or formatting diff | Run `pre-commit run --all-files` locally to auto-fix ruff and ruff-format issues |

**3. Read the workflow logs**

In GitHub: Actions tab → failing workflow run → click the failing job → expand the failing step.
The full `pytest` output including tracebacks is in the "Run tests" step. Coverage XML is uploaded
as an artifact for download.

For flaky failures (rare import ordering issues with `pytest-forked`), re-run the failed job
directly from the Actions UI — Dagster import isolation is the most common source of transient
failures.

---

## How to update DVC-tracked data

**After re-running the feature engineering pipeline:**

```bash
# Re-materialise features (replaces parquets in data/processed/features/)
shelfsense materialize --asset features

# Update the DVC pointer file and push
dvc add data/processed/features/
git add data/processed/features.dvc data/processed/.gitignore
git commit -m "data: update feature parquets"
dvc push
```

**After downloading fresh raw M5 data:**

```bash
dvc add data/raw/m5-forecasting-accuracy/
git add data/raw/m5-forecasting-accuracy.dvc data/raw/.gitignore
git commit -m "data: update raw M5 CSVs"
dvc push
```

**To update the remote folder URL:**

```bash
dvc remote modify gdrive url gdrive://<new-folder-id>
git add .dvc/config
git commit -m "dvc: update gdrive remote folder"
```

---

## How to reproduce a specific past experiment from MLflow

Every `shelfsense train` or `shelfsense materialize` run logs params, metrics, and tags to MLflow.
To reproduce a specific run:

**1. Open the MLflow UI and find the run**

```bash
mlflow ui --port 5000 --backend-store-uri sqlite:///mlflow.db
# Open http://localhost:5000 in a browser
# Navigate to the ShelfSense experiment and find the run by name and date
```

**2. Collect the run parameters**

From the MLflow UI "Parameters" tab (or CLI):

```bash
mlflow runs get --run-id <run-id>
```

Key params: `objective`, `tvp`, `test_mode`. The `asset` tag identifies which Dagster asset was
materialised.

**3. Check out the matching git commit**

MLflow tags include `git_commit` if `shelfsense.tracking.mlflow_utils` was configured to log it.
If not, use the run timestamp to identify the commit:

```bash
git log --after="<run-date minus 1 hour>" --before="<run-date plus 1 hour>" --oneline
```

**4. Restore the matching data version**

```bash
git checkout <commit>
dvc checkout       # restores data/processed/features/ to the version tracked at that commit
```

**5. Re-run the asset**

```bash
shelfsense materialize --asset model_tvp_13   # or whichever variant the run used
```

The Hydra config is read from `shelfsense/config/model/` at runtime — as long as the config files
match the checked-out commit, the run is reproducible bit-for-bit (all seeds are in the YAML).

---

## Known constraints

**Dagster pinned to `<1.9.3` — antlr4 conflict with hydra-core**

`dagster>=1.9.3` ships `AssetSelectionLexer.py` generated by ANTLR 4.13.2, whose ATN serialization
format (v4) is incompatible with `antlr4-python3-runtime==4.9.3` required by `hydra-core==1.3.2`
and `omegaconf==2.3.0`. Dagster 1.9.2 is the latest version without this conflict; it uses a
regex-based `parse_clause()` for asset selection. **Resolution path:** drop `hydra-core` and the
`antlr4==4.9.*` pin, then upgrade Dagster freely.

In Dagster 1.9.2, `*asset_name` means "asset and all transitive upstreams" — equivalent to
`+asset_name` in the ANTLR DSL of 1.9.3+.

**WSL2 memory pressure on full pipeline materialisation**

Sequential `dagster.materialize()` of all 22 assets keeps DataFrames alive across step boundaries,
which can exceed 8 GB on WSL2 (default 50% host RAM cap). Symptom: kernel OOM-kills the process
during the features → model transition. Workarounds:
- Allocate `>16 GB` to WSL2 in `.wslconfig`
- Use `--asset` to materialise in smaller batches (e.g., `features` first, then each model)
- Long-term fix: adopt a Dagster IO manager that serialises each asset output to disk so
  intermediate DataFrames are freed between steps.

**GitHub Actions: Node.js 20 deprecation**

CI warns: "Node.js 20 actions are deprecated … forced to run with Node.js 24 by default starting
June 2nd, 2026." Affected actions: `actions/checkout@v4`, `actions/setup-python@v5`,
`actions/cache@v4`, `astral-sh/setup-uv@v3`. Upgrade each to its latest Node.js 24 compatible
release before the June 2026 deadline. No action needed today — workflows still pass on Node.js 20.

---

## Lessons learned — Stage 3 dependency gap (commit 22)

**What happened.** Stage 3 added `pandera` and `dvc` but placed both in
`[project.optional-dependencies] dev`. `pandera` is imported at module level by
`shelfsense/data/schemas.py` — it is a runtime dependency, not a test tool. Tests always ran
against the dev venv (which had `--extra dev`), so pandera was silently present and every test
passed. The gap became visible when imagining a fresh `pip install shelfsense` or a production
`uv sync --no-dev`: the import would fail at startup.

**Rule.** Before closing any stage, verify the dependency contract from the consumer's perspective:

1. `uv run --no-dev python -c "from shelfsense.<module> import <symbol>"` for every new public
   module. Failure means the dep belongs in `[project.dependencies]`.
2. `uv sync --frozen --no-editable` (the Docker production path) must install every import the
   package makes. Dev extras are not installed in the production image.
3. CLI-only tools that are never imported by package code (e.g., `dvc`) correctly stay in dev
   extras.
