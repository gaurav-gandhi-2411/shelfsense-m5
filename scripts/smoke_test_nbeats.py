"""
N-BEATS smoke test: 100 series, 3 epochs x 30 steps, GPU.
Verifies NaN fix (d_num>=181 clip + price ffill) and VRAM policy.

IMPORT ORDER MATTERS on Windows:
  Load all data (parquet → TimeSeries) BEFORE initializing CUDA.
  torch.set_float32_matmul_precision() and cap_vram() both init the CUDA
  context. If called before pd.read_parquet(), coreforecast/numba's LLVM
  conflicts with PyTorch's LLVM → segfault.
"""
import sys, time, gc
sys.path.insert(0, '.')
import warnings; warnings.filterwarnings('ignore')
import logging
logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)
logging.getLogger('lightning').setLevel(logging.ERROR)

# --- Step 1: darts + adapter imports (torch imported lazily, CUDA not yet live) ---
from src.shelfsense.data.dl_format_adapter import to_long_format, to_darts_datasets
from darts.models import NBEATSModel

# --- Step 2: load and convert data (must be BEFORE CUDA init) ---
t0 = time.time()
df = to_long_format('data/processed/features/store_CA_4.parquet')
print(f'Loaded in {time.time()-t0:.1f}s: shape={df.shape}')
print(f'  ds range: {df["ds"].min().date()} to {df["ds"].max().date()}')
print(f'  rows/series: {df.groupby("unique_id").size().unique().tolist()}')

uids = df['unique_id'].unique()[:100]
df_small = df[df['unique_id'].isin(uids)].copy()
del df; gc.collect()

t1 = time.time()
targets, past_covs, _ = to_darts_datasets(df_small)
del df_small; gc.collect()
print(f'TimeSeries built in {time.time()-t1:.1f}s: {len(targets)} series x {len(targets[0])} days')

# --- Step 3: NOW initialize CUDA (after all data is loaded) ---
import torch
torch.set_float32_matmul_precision('medium')
from src.shelfsense.data.vram_utils import cap_vram, vram_status
cap_vram(0.80)
print(f'CUDA init OK | {vram_status()}')

# --- Step 4: train ---
print()
print('Training N-BEATS (3 epochs x 30 steps, batch=16, GPU, cap=6.4GB)...')
model = NBEATSModel(
    input_chunk_length=56,
    output_chunk_length=28,
    n_epochs=3,
    batch_size=16,
    pl_trainer_kwargs={
        'accelerator': 'gpu',
        'devices': 1,
        'enable_progress_bar': True,
        'limit_train_batches': 30,
    },
    random_state=42,
)

t2 = time.time()
model.fit(targets, past_covariates=past_covs, verbose=True)
elapsed = time.time() - t2
print(f'Done in {elapsed:.1f}s ({elapsed/3:.1f}s/epoch)')
print(f'VRAM after train: {vram_status()}')

# --- Step 5: predict ---
print()
print('Predicting (3 series, horizon=28)...')
preds = model.predict(n=28, series=targets[:3], past_covariates=past_covs[:3])
print('Pred shape:', preds[0].to_series().shape)
print('Sample (series 0, days 1-5):', preds[0].to_series().values[:5].round(3))
print('Any NaN in preds:', any(p.to_series().isna().any() for p in preds))

print()
print('SMOKE TEST PASSED')
