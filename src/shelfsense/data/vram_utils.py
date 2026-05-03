"""VRAM management utilities for WS2 training on RTX 3070 (8GB).

Policy: reserve 1.5-2 GB for system (display, browser, tools).
Train within 6-6.5 GB. Hard cap via set_per_process_memory_fraction(0.80).

WINDOWS IMPORT ORDER CONSTRAINT:
  Call cap_vram() and set_float32_matmul_precision() AFTER all data loading
  (pd.read_parquet, TimeSeries construction). On Windows, initializing the CUDA
  context before coreforecast/numba's LLVM loads causes a segfault due to
  PyTorch and numba shipping separate LLVM builds that conflict when both are
  live simultaneously. Safe pattern in every training script:

    from darts.models import NBEATSModel        # torch imported lazily, no CUDA yet
    df = to_long_format(...)                    # parquet + numba JIT safe
    targets = to_darts_datasets(df)             # TimeSeries construction safe
    import torch; torch.set_float32_matmul_precision('medium')  # NOW init CUDA
    cap_vram(0.80)                              # hard cap after CUDA is live
    model.fit(...)                              # GPU training
"""
import torch


def cap_vram(fraction: float = 0.80) -> None:
    """Hard-cap this process to `fraction` of total GPU VRAM.

    Call once at the top of any training script, before model creation.
    0.80 × 8 GB = 6.4 GB — leaves ~1.6 GB for system/display.
    """
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(fraction, device=0)


def assert_vram_headroom(min_free_gb: float = 1.5) -> None:
    """Raise if free VRAM falls below min_free_gb.

    Call before large allocations (model creation, batch size experiments).
    Does nothing when CUDA is unavailable (CPU-only environments).
    """
    if not torch.cuda.is_available():
        return
    free, total = torch.cuda.mem_get_info(device=0)
    free_gb = free / 1e9
    total_gb = total / 1e9
    if free_gb < min_free_gb:
        raise RuntimeError(
            f"VRAM headroom violation: {free_gb:.2f} GB free of {total_gb:.1f} GB total "
            f"(need {min_free_gb} GB). Reduce batch_size or free GPU memory first."
        )


def vram_status() -> str:
    """Return a one-line VRAM status string for logging."""
    if not torch.cuda.is_available():
        return "CUDA unavailable"
    free, total = torch.cuda.mem_get_info(device=0)
    used_gb = (total - free) / 1e9
    total_gb = total / 1e9
    return f"VRAM {used_gb:.2f}/{total_gb:.1f} GB used ({100*(total-free)/total:.0f}%)"
