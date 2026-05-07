"""
Rolling statistics for M5.

Input:  long-format DataFrame sorted by (id, d_num), containing 'id', 'sales'.
Output: same DataFrame with 16 rolling columns appended (float32).

Features: roll_{stat}_{window} for stat in {mean, std, min, max}, window in {7, 28, 56, 180}.

Leakage note: sales is shifted by 1 day before rolling so the most recent observed
value at day d is d-1.  Rolling window [d-w-1 .. d-1] -> no future leakage.
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
    Append rolling statistic columns to df in-place and return df.

    Produces roll_{stat}_{window} for each stat in stats and each window in
    windows (up to 4 × 4 = 16 columns with defaults). Sales is shifted by 1
    day before aggregation to prevent same-day leakage.

    Parameters
    ----------
    df :
        Long-format DataFrame sorted by (id_col, d_num). Must contain id_col
        and sales_col. Sorting is the caller's responsibility.
    windows :
        Rolling window sizes in days. Default: [7, 28, 56, 180].
    stats :
        Aggregation functions to apply. Supported: 'mean', 'std', 'min', 'max'.
    sales_col :
        Name of the daily sales column.
    id_col :
        Name of the series identifier column.

    Returns
    -------
    pd.DataFrame
        Same object as df with roll_{stat}_{window} columns appended as float32.
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
