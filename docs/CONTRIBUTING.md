# Contributing to ShelfSense M5

## Setup

```bash
# Clone and install in dev mode
git clone https://github.com/<you>/shelfsense-m5.git
cd shelfsense-m5
uv sync --extra dev

# Pull M5 data (requires DVC + service account key — see docs/RUNBOOK.md)
dvc pull

# Install pre-commit hooks
pre-commit install
```

## Pre-commit hooks

The repo uses [pre-commit](https://pre-commit.com/) for automated checks on every `git commit`:

| Hook | What it checks |
|---|---|
| `trailing-whitespace` | No trailing spaces |
| `end-of-file-fixer` | Files end with a newline |
| `check-yaml` / `check-toml` | Config files parse cleanly |
| `ruff` | Lint + auto-fix (E, F, I rules) |
| `ruff-format` | Code formatting |
| `dvc-status` | Warns if DVC-tracked data was modified without updating `.dvc` files |
| `shelfsense-data-validate` | Runs Pandera schema checks on raw CSVs and feature parquets (skipped when data absent) |

Run all hooks manually: `pre-commit run --all-files`

### DVC hook behaviour

`dvc-status` runs in warn-only mode (always exits 0). If it reports modified
data, update the pointer files before committing:

```bash
dvc add data/processed/features/
git add data/processed/features.dvc
```

### Data validate hook behaviour

`shelfsense-data-validate` fires only when `.dvc` pointer files or
`shelfsense/data/` source files are staged. On machines without the data it
is a no-op. On machines with data it exits 1 if any schema check fails,
blocking the commit — fix the data issue or update the schema before
proceeding.

## Running tests

```bash
make test        # pytest --cov with term-missing report
make lint        # ruff + black --check + mypy
make format      # ruff --fix + black (auto-formats)
```

## Commit conventions

- Prefix: `stage1:` / `stage2:` / `stage3:` / `feat:` / `fix:` / `data:` / `docs:`
- Keep commits focused: one logical change per commit
- Tests must pass before every commit (`make test`)
