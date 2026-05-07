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
