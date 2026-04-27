"""
Calendar feature builder for M5.

build_calendar_lookup(calendar_df) -> pd.DataFrame indexed by 'd' (e.g. 'd_1')
    Returns one row per calendar day with date/event/SNAP features.

add_calendar_features(df, cal_lookup) -> df
    Joins calendar features onto long-format sales DataFrame via 'd' column.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_calendar_lookup(calendar_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a day-keyed lookup table of calendar features.

    Columns produced:
      date, d_num, weekday (0=Mon..6=Sun), month, quarter, year,
      day_of_month, week_of_year, is_weekend, is_holiday,
      is_snap_ca, is_snap_tx, is_snap_wi,
      days_since_event, days_until_next_event
    """
    cal = calendar_df.copy()
    cal["date"] = pd.to_datetime(cal["date"])

    # integer day index
    cal["d_num"] = cal["d"].str.replace("d_", "", regex=False).astype(np.int16)

    # weekday: pandas dt.dayofweek = 0 (Mon) .. 6 (Sun)
    cal["weekday"]      = cal["date"].dt.dayofweek.astype(np.int8)
    cal["month"]        = cal["date"].dt.month.astype(np.int8)
    cal["quarter"]      = cal["date"].dt.quarter.astype(np.int8)
    cal["year"]         = cal["date"].dt.year.astype(np.int16)
    cal["day_of_month"] = cal["date"].dt.day.astype(np.int8)
    cal["week_of_year"] = cal["date"].dt.isocalendar().week.astype(np.int8)
    cal["is_weekend"]   = cal["weekday"].isin([5, 6]).astype(np.int8)

    # holiday: either event slot filled
    cal["is_holiday"] = (
        cal["event_name_1"].notna() | cal["event_name_2"].notna()
    ).astype(np.int8)

    cal["is_snap_ca"] = cal["snap_CA"].astype(np.int8)
    cal["is_snap_tx"] = cal["snap_TX"].astype(np.int8)
    cal["is_snap_wi"] = cal["snap_WI"].astype(np.int8)

    # days_since_event / days_until_next_event (based on event_name_1 only)
    event_days = cal.loc[cal["event_name_1"].notna(), "d_num"].values
    d_nums = cal["d_num"].values

    # searchsorted gives the insertion index into the sorted event_days array
    idx_after = np.searchsorted(event_days, d_nums, side="right")   # first event > d
    idx_prev  = idx_after - 1                                         # last event <= d

    # days_since: d_num - last_event_day (NaN if no prior event)
    days_since = np.where(
        idx_prev >= 0,
        d_nums - event_days[np.clip(idx_prev, 0, len(event_days) - 1)],
        np.nan,
    )
    # clip to 0 (same-day event → 0)
    days_since = np.where(days_since < 0, np.nan, days_since)

    # days_until: next_event_day - d_num (NaN if no future event)
    days_until = np.where(
        idx_after < len(event_days),
        event_days[np.clip(idx_after, 0, len(event_days) - 1)] - d_nums,
        np.nan,
    )
    days_until = np.where(days_until < 0, np.nan, days_until)

    cal["days_since_event"]      = days_since.astype(np.float32)
    cal["days_until_next_event"] = days_until.astype(np.float32)

    keep = [
        "d", "d_num", "weekday", "month", "quarter", "year",
        "day_of_month", "week_of_year", "is_weekend", "is_holiday",
        "is_snap_ca", "is_snap_tx", "is_snap_wi",
        "days_since_event", "days_until_next_event",
        "wm_yr_wk",   # kept for price join downstream
    ]
    return cal[keep].set_index("d")


def add_calendar_features(df: pd.DataFrame, cal_lookup: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join calendar features onto df using the 'd' column.
    Returns df with calendar columns appended.
    """
    return df.join(cal_lookup, on="d")
