# Contributing to ShelfSense M5

## Setting up the dev environment

```bash
# 1. Clone and create the virtual environment
git clone https://github.com/gaurav-gandhi-2411/shelfsense-m5.git
cd shelfsense-m5

# 2. Install all dependencies including dev extras (uv creates the venv automatically)
uv sync --extra dev

# 3. Activate the environment
source .venv/bin/activate          # Linux / macOS / WSL2
# .venv\Scripts\activate           # Windows PowerShell

# 4. Install the pre-commit hooks
pre-commit install

# 5. Pull M5 data (requires DVC + Google Drive service account — see docs/RUNBOOK.md)
dvc pull
```

Verify the installation:

```bash
shelfsense --help      # should list: data, features, train, ensemble, submit, materialize, report
make test              # all unit tests green
```

---

## Running tests

```bash
make test              # pytest with coverage report (fails below 60%)
make lint              # ruff check + mypy
make format            # ruff --fix + ruff-format (auto-formats in-place)
```

Direct pytest commands when you want finer control:

```bash
# Fast: unit tests only, stop on first failure
uv run pytest tests/unit/ -x

# With coverage detail
uv run pytest tests/ --cov=shelfsense --cov-report=term-missing

# Single test file
uv run pytest tests/unit/test_cli.py -v

# Integration tests (requires M5 data — slower, ~5 min)
uv run pytest tests/integration/ -v
```

Tests run with `pytest-forked` by default (`-p forked` in `pyproject.toml`) to isolate Dagster
import state between tests. Pass `-p no:forked` for faster iteration during development when you
don't need isolation.

---

## Pre-commit hooks

The repo uses [pre-commit](https://pre-commit.com/) for automated checks on every `git commit`.

| Hook | What it checks |
|------|----------------|
| `trailing-whitespace` | No trailing spaces |
| `end-of-file-fixer` | Files end with a newline |
| `check-yaml` / `check-toml` | Config files parse cleanly |
| `check-merge-conflict` | No unresolved merge markers |
| `debug-statements` | No leftover `breakpoint()` / `pdb` calls |
| `check-added-large-files` | Blocks files >500 KB from being committed |
| `ruff` | Lint + auto-fix (E, F, I, W rules) |
| `ruff-format` | Code formatting (authoritative formatter) |
| `mypy` | Type checking (`shelfsense/` only, lenient — `--allow-untyped-defs`) |
| `dvc-status` | Warns if DVC-tracked data was modified without updating `.dvc` files |

Run all hooks manually: `pre-commit run --all-files`

The `dvc-status` hook always exits 0 — it is informational only. If it reports modified data,
update the pointer files before committing (see `docs/RUNBOOK.md` → "How to update DVC-tracked
data").

To skip hooks for a single commit when you have a known reason (e.g., committing a work-in-progress
during an investigation):

```bash
SKIP=mypy git commit -m "wip: ..."    # skip a specific hook
git commit --no-verify                 # skip all hooks (use sparingly)
```

---

## Commit message conventions

Format: `<prefix>: <verb> <what>`

Prefixes used in this repo:

| Prefix | When to use |
|--------|-------------|
| `stage1:` … `stage6:` | Work within a numbered Phase 3 stage |
| `feat:` | New capability outside a stage (model variant, CLI command) |
| `fix:` | Bug fix |
| `data:` | DVC pointer update, raw data change |
| `docs:` | Documentation-only change |
| `tests:` | Test-only change |
| `cleanup:` | Dead code removal, archive, dep pinning |
| `results:` | Score recording after a Kaggle submission |

Examples from the actual git log:

```
stage6: add docs/MODELS.md (commit 37)
stage5: pre-commit hooks for ruff, ruff-format, mypy, dvc-status
stage4: resolve antlr4 conflict between hydra-core and dagster
fix: update per-store/per-dept stub tests to mocked; add data validate test for coverage
results: annual lags private LB 0.5749 — Phase 2 closed
cleanup: archive unsubmitted CSV, pin deps, delete dead code, launch ylags training
```

Keep commits focused: one logical change per commit. Tests must pass before every commit
(`make test`). Avoid `git commit --no-verify` except during active debugging — hooks are fast
(< 10 seconds on the `shelfsense/` package).

---

## Pull request conventions

This is a single-developer project, but PRs are used for non-trivial feature branches to preserve
a reviewable record.

**Title.** Match the commit prefix format: `stage6: documentation polish` or `feat: add cv_evaluation asset`.

**Body.** Three sections:

```markdown
## What
One sentence: what changed.

## Why
One sentence: why this change was needed or what problem it solves.

## Test plan
- [ ] `make test` passes
- [ ] `pre-commit run --all-files` passes
- [ ] (if relevant) `shelfsense materialize --asset <name>` completes without error
```

**Branch naming.** `stage<N>/<short-slug>` or `fix/<short-slug>`. Delete the branch after merge.
