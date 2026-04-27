"""
Lag features for M5.

Input:  long-format DataFrame sorted by (id, d_num), containing 'id', 'd_num', 'sales'.
Output: same DataFrame with lag_7, lag_14, lag_28, lag_56 columns appended (float32).

Leakage note: lag_N at day d = sales at day (d - N).  Requires df sorted by (id, d_num).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LAGS = [7, 14, 28, 56]


def add_lags(
    df: pd.DataFrame,
    lags: list[int] = LAGS,
    sales_col: str = "sales",
    id_col: str = "id",
) -> pd.DataFrame:
    """
    Add lag features in-place.  df must be sorted by (id, d_num).
    """
    for lag in lags:
        df[f"lag_{lag}"] = (
            df.groupby(id_col)[sales_col]
            .shift(lag)
            .astype(np.float32)
        )
    return df
