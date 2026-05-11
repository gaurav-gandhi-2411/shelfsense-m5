"""M5Dataset — validated loader for raw M5 CSVs and processed feature parquets."""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from shelfsense.data.schemas import (
    feature_schema,
    raw_calendar_schema,
    raw_prices_schema,
    raw_sales_schema,
)


@dataclass
class M5Dataset:
    """Thin wrapper around the M5 data files that validates on load.

    Parameters
    ----------
    raw_dir:
        Path to directory containing the raw M5 CSVs
        (sales_train_evaluation.csv, calendar.csv, sell_prices.csv).
    features_dir:
        Path to directory containing per-store feature parquets.
    validate:
        If True, run Pandera schema checks on each file as it is loaded.
        Raises SchemaError / SchemaErrors on violation.
    """

    raw_dir: str
    features_dir: str
    validate: bool = True

    _sales: Optional[pd.DataFrame] = field(default=None, repr=False, init=False)
    _calendar: Optional[pd.DataFrame] = field(default=None, repr=False, init=False)
    _prices: Optional[pd.DataFrame] = field(default=None, repr=False, init=False)

    # ------------------------------------------------------------------
    # Raw CSVs
    # ------------------------------------------------------------------

    @property
    def sales(self) -> pd.DataFrame:
        if self._sales is None:
            path = os.path.join(self.raw_dir, "sales_train_evaluation.csv")
            df = pd.read_csv(path)
            if self.validate:
                raw_sales_schema.validate(df, lazy=True)
            self._sales = df
        return self._sales

    @property
    def calendar(self) -> pd.DataFrame:
        if self._calendar is None:
            path = os.path.join(self.raw_dir, "calendar.csv")
            df = pd.read_csv(path)
            if self.validate:
                raw_calendar_schema.validate(df, lazy=True)
            self._calendar = df
        return self._calendar

    @property
    def prices(self) -> pd.DataFrame:
        if self._prices is None:
            path = os.path.join(self.raw_dir, "sell_prices.csv")
            df = pd.read_csv(path)
            if self.validate:
                raw_prices_schema.validate(df, lazy=True)
            self._prices = df
        return self._prices

    # ------------------------------------------------------------------
    # Feature parquets
    # ------------------------------------------------------------------

    def feature_paths(self) -> list[str]:
        pattern = os.path.join(self.features_dir, "*.parquet")
        return sorted(glob.glob(pattern))

    def load_features(self, store: Optional[str] = None) -> pd.DataFrame:
        """Load feature parquets and return a concatenated DataFrame.

        Parameters
        ----------
        store:
            If given, load only the parquet for that store_id
            (e.g. ``"CA_1"``). Otherwise load and concat all stores.
        """
        if store is not None:
            path = os.path.join(self.features_dir, f"{store}.parquet")
            df = pd.read_parquet(path, engine="pyarrow")
            if self.validate:
                feature_schema.validate(df, lazy=True)
            return df

        parts: list[pd.DataFrame] = []
        for path in self.feature_paths():
            df = pd.read_parquet(path, engine="pyarrow")
            if self.validate:
                feature_schema.validate(df, lazy=True)
            parts.append(df)
        return pd.concat(parts, ignore_index=True)

    # ------------------------------------------------------------------
    # Batch validation helpers (used by CLI)
    # ------------------------------------------------------------------

    def validate_raw(self) -> dict[str, bool | str]:
        """Validate all three raw CSVs. Returns {filename: passed}."""
        results: dict[str, bool | str] = {}
        for attr, fname in (
            ("sales", "sales_train_evaluation.csv"),
            ("calendar", "calendar.csv"),
            ("prices", "sell_prices.csv"),
        ):
            try:
                getattr(self, attr)
                results[fname] = True
            except Exception as exc:  # noqa: BLE001
                results[fname] = False
                results[f"{fname}__error"] = str(exc)
        return results

    def validate_features(self) -> dict[str, bool | str]:
        """Validate each feature parquet. Returns {filename: passed}."""
        results: dict[str, bool | str] = {}
        for path in self.feature_paths():
            fname = os.path.basename(path)
            try:
                df = pd.read_parquet(path, engine="pyarrow")
                feature_schema.validate(df, lazy=True)
                results[fname] = True
            except Exception as exc:  # noqa: BLE001
                results[fname] = False
                results[f"{fname}__error"] = str(exc)
        return results
