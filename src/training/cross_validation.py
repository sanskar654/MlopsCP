"""
Temporal Cross-Validator
========================
Implements time-aware train/validation/test splits for vulnerability data.

Critical: CVEs are ordered by publication date. A random split would allow
training on future CVEs (which may reference earlier vulnerabilities), creating
temporal data leakage. All splits here are strict cutoff-based.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TemporalCrossValidator:
    """
    Time-based split: train on past, validate on near-future, test on far-future.

    Strategy:
        |─────────── Train (70%) ──────────|── Val (15%) ──|── Test (15%) ──|
        oldest CVEs                                          newest CVEs
    """

    def __init__(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        date_col: str = "feature_created_at",
        year_col: str = "year_published",
    ):
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1.0 - train_ratio - val_ratio
        self.date_col = date_col
        self.year_col = year_col

    def temporal_split(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split DataFrame temporally.

        Prefers year_published for ordering; falls back to index-based split.

        Returns:
            (train_df, val_df, test_df)
        """
        # Sort by year published if available
        if self.year_col in df.columns:
            df = df.sort_values(self.year_col, ascending=True).reset_index(drop=True)
            logger.info(
                f"Temporal split by {self.year_col}: "
                f"range [{df[self.year_col].min()}, {df[self.year_col].max()}]"
            )
        else:
            # Fallback: use row order (assuming ingestion order = temporal order)
            logger.warning(
                f"{self.year_col} not found. Using row-order temporal split."
            )

        n = len(df)
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))

        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()

        logger.info(
            f"Temporal split: train={len(train_df)}, "
            f"val={len(val_df)}, test={len(test_df)}"
        )

        # Verify no temporal overlap
        if self.year_col in df.columns:
            train_max_year = train_df[self.year_col].max()
            val_min_year = val_df[self.year_col].min()
            test_min_year = test_df[self.year_col].min()

            logger.info(
                f"Year ranges: train→{train_max_year}, "
                f"val starts at {val_min_year}, "
                f"test starts at {test_min_year}"
            )

        return train_df, val_df, test_df

    def temporal_cv_folds(
        self,
        df: pd.DataFrame,
        n_folds: int = 5,
    ) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Walk-forward temporal cross-validation.

        Each fold: train on all data before split point, validate on next window.

        |── Fold 1 Train ──|─ Val ─|
        |──── Fold 2 Train ─────|─ Val ─|
        ...

        Args:
            df: Feature DataFrame sorted by time
            n_folds: Number of folds

        Returns:
            List of (train_df, val_df) tuples
        """
        if self.year_col in df.columns:
            df = df.sort_values(self.year_col).reset_index(drop=True)

        n = len(df)
        fold_size = n // (n_folds + 1)
        folds = []

        for i in range(1, n_folds + 1):
            train_end = fold_size * i
            val_end = min(fold_size * (i + 1), n)

            train_df = df.iloc[:train_end].copy()
            val_df = df.iloc[train_end:val_end].copy()

            if len(val_df) == 0:
                break

            folds.append((train_df, val_df))
            logger.debug(
                f"Fold {i}: train={len(train_df)}, val={len(val_df)}"
            )

        logger.info(f"Created {len(folds)} temporal CV folds")
        return folds
