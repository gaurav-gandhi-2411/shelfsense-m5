"""
Rolling statistics for M5.

Input:  long-format DataFrame sorted by (id, d_num), containing 'id', 'sales'.
Output: same DataFrame with 16 rolling columns appended (float32).

Features: roll_{stat}_{window} for stat in {mean, std, min, max}, window in {7, 28, 56, 180}.

Leakage note: sales is shifted by 1 day before rolling so the most recent observed
value at day d is d-1.  Rolling window [d-w-1 .. d-1] → no future leakage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WINDOWS = [7, 28, 56, 180]
STATS = ["mean", "std", "min", "max"]


def add_rolling(
    df: pd.DataFrame,
    windows: list[int] = WINDOWS,
    stats: list[str] = STATS,
    sales_col: str = "sales",
    id_col: str = "id",
) -> pd.DataFrame:
    """
    Add rolling features in-place.  df must be sorted by (id, d_num).

    Uses a single shift-1 pass then applies rolling aggregations to avoid
    recomputing the shift for each window.
    """
    shifted = df.groupby(id_col)[sales_col].shift(1)

    for window in windows:
        roll = shifted.groupby(df[id_col]).rolling(window, min_periods=1)
        if "mean" in stats:
            df[f"roll_mean_{window}"] = roll.mean().reset_index(level=0, drop=True).astype(np.float32)
        if "std" in stats:
            df[f"roll_std_{window}"] = roll.std(ddof=1).reset_index(level=0, drop=True).astype(np.float32)
        if "min" in stats:
            df[f"roll_min_{window}"] = roll.min().reset_index(level=0, drop=True).astype(np.float32)
        if "max" in stats:
            df[f"roll_max_{window}"] = roll.max().reset_index(level=0, drop=True).astype(np.float32)

    return df
