"""
Hierarchy categorical features for M5.

LightGBM accepts category dtype natively — no one-hot encoding needed.
This module just casts the four hierarchy columns to pd.CategoricalDtype
with the full known value set so that all partitions share the same encoding.
"""
from __future__ import annotations

import pandas as pd

# Fixed category levels derived from M5 dataset (deterministic, no need to infer)
CAT_LEVELS = {
    "cat_id":   ["FOODS", "HOUSEHOLD", "HOBBIES"],
    "dept_id":  [
        "FOODS_1", "FOODS_2", "FOODS_3",
        "HOUSEHOLD_1", "HOUSEHOLD_2",
        "HOBBIES_1", "HOBBIES_2",
    ],
    "store_id": [
        "CA_1", "CA_2", "CA_3", "CA_4",
        "TX_1", "TX_2", "TX_3",
        "WI_1", "WI_2", "WI_3",
    ],
    "state_id": ["CA", "TX", "WI"],
}

# LightGBM needs integer codes; we expose the ordered dtype so every partition
# uses the same code mapping.
CAT_DTYPES = {
    col: pd.CategoricalDtype(categories=cats, ordered=False)
    for col, cats in CAT_LEVELS.items()
}


def add_hierarchy_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cast cat_id, dept_id, store_id, state_id to categorical dtype in-place.
    Columns must already exist in df (they come from the melt + meta join).
    """
    for col, dtype in CAT_DTYPES.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)
    return df
