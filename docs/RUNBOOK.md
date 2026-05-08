# ShelfSense M5 — Operations Runbook

## DVC Remote Setup (Google Drive)

DVC is configured with a Google Drive remote named `gdrive`. The URL uses a
folder path token; authentication uses a GCP service account so `dvc push/pull`
works non-interactively in CI and fresh clones.

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

Open the Drive folder whose ID is in `.dvc/config` (the `gdrive://` URL) and
share it with the service account email:
`shelfsense-dvc@<PROJECT_ID>.iam.gserviceaccount.com` — Editor access.

**4. Configure the local DVC credential path**

```bash
dvc remote modify --local gdrive gdrive_use_service_account true
dvc remote modify --local gdrive gdrive_service_account_json_file_path secrets/gdrive-sa.json
```

These are written to `.dvc/config.local` which is gitignored by DVC.

**5. Push data to remote**

```bash
dvc push          # uploads data/raw/m5-forecasting-accuracy/ and data/processed/features/
```

**6. Pull data on a new machine**

```bash
# Clone repo, complete steps 2-4, then:
dvc pull
```

### Updating tracked data

After rerunning the feature engineering pipeline:

```bash
dvc add data/processed/features/
git add data/processed/features.dvc data/processed/.gitignore
git commit -m "data: update feature parquets"
dvc push
```

### Remote URL

The `gdrive://` folder ID in `.dvc/config` must match the Google Drive folder
shared with the service account. To update it:

```bash
dvc remote modify gdrive url gdrive://<new-folder-id>
git add .dvc/config
git commit -m "dvc: update gdrive remote folder"
```

---

## Lessons learned — Stage 3 dependency gap (commit 22)

**What happened:** Stage 3 added `pandera` (commits 19-20) and `dvc` (commit
17) but placed both in `[project.optional-dependencies] dev` instead of their
correct locations. `pandera` is imported at module level by
`shelfsense/data/schemas.py` and `shelfsense/data/load.py` — it is a runtime
dependency of the package, not a test tool. Tests were always run via
`uv run pytest` against the dev venv (which had `--extra dev` installed), so
pandera was silently present and every test passed. The gap only became visible
when imagining a fresh `pip install shelfsense` or a Docker `uv sync` run
without dev extras: the import would fail at container startup.

**Rule derived:** Before declaring a stage complete, verify the dependency
contract from the consumer's perspective, not the developer's:

1. Run `uv run --no-dev python3 -c "from shelfsense.<module> import <symbol>"`
   for every new public module. If it fails, the dep belongs in
   `[project.dependencies]`, not in an extra.
2. Check that `uv sync --frozen --no-editable` (the Docker production path)
   would install every import the package makes. Dev extras are NOT installed
   in the production image.
3. `dvc` and other CLI-only tools that are never imported by package code
   correctly stay in dev extras — they are not needed at container runtime.
