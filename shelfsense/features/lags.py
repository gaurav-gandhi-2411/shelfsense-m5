"""
Lag features for M5.

Input:  long-format DataFrame sorted by (id, d_num), containing 'id', 'd_num', 'sales'.
Output: same DataFrame with lag columns appended (float32).

Leakage note: lag_N at day d = sales at day (d - N).  Requires df sorted by (id, d_num).

lag_91/182/364 are always populated for d_num >= 365. For training at d_num >= 1000
(FEAT_START) all three are fully available. They will be NaN for d_num < 365 in the
full parquet — expected and benign (those rows are excluded at training time).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LAGS = [7, 14, 28, 56, 91, 182, 364]


def add_lags(
    df: pd.DataFrame,
    lags: list[int] = LAGS,
    sales_col: str = "sales",
    id_col: str = "id",
) -> pd.DataFrame:
    """
    Append lag_N columns to df in-place and return df.

    Parameters
    ----------
    df :
        Long-format DataFrame sorted by (id_col, d_num). Must contain id_col
        and sales_col. Sorting is the caller's responsibility.
    lags :
        Lag periods in days. Each period N produces a column lag_N
        containing sales[d - N] for each series. Default: [7, 14, 28, 56, 91, 182, 364].
    sales_col :
        Name of the daily sales column.
    id_col :
        Name of the series identifier column.

    Returns
    -------
    pd.DataFrame
        Same object as df with lag_{N} columns appended as float32.
        NaN where history is insufficient (first N days of each series).
    """
    for lag in lags:
        df[f"lag_{lag}"] = (
            df.groupby(id_col)[sales_col]
            .shift(lag)
            .astype(np.float32)
        )
    return df
