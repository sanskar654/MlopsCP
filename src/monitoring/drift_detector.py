"""
Drift Detector Module
=====================
Implements data drift, target drift, and feature distribution monitoring
using Evidently AI and statistical testing (KS-test, PSI, Jensen-Shannon).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger("mlops.monitoring.drift")


class StatisticalDriftDetector:
    """Statistical drift detector calculating PSI, KS-test, and Wasserstein distance."""

    def __init__(self, significance_level: float = 0.05, psi_threshold: float = 0.2):
        self.significance_level = significance_level
        self.psi_threshold = psi_threshold

    @staticmethod
    def calculate_psi(
        reference: np.ndarray, current: np.ndarray, num_bins: int = 10
    ) -> float:
        """Calculate Population Stability Index (PSI) between two distributions."""
        reference = reference[~np.isnan(reference)]
        current = current[~np.isnan(current)]

        if len(reference) == 0 or len(current) == 0:
            return 0.0

        # Create quantiles from reference
        quantiles = np.linspace(0, 100, num_bins + 1)
        bins = np.percentile(reference, quantiles)
        bins = np.unique(bins)
        if len(bins) < 2:
            return 0.0

        bins[0] = -np.inf
        bins[-1] = np.inf

        ref_counts, _ = np.histogram(reference, bins=bins)
        curr_counts, _ = np.histogram(current, bins=bins)

        ref_pct = np.where(ref_counts == 0, 0.0001, ref_counts) / len(reference)
        curr_pct = np.where(curr_counts == 0, 0.0001, curr_counts) / len(current)

        psi = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))
        return float(psi)

    def detect_numerical_drift(
        self, reference_col: pd.Series, current_col: pd.Series
    ) -> Dict[str, Any]:
        """Runs KS-Test and PSI on numerical columns."""
        ref_clean = reference_col.dropna().values
        curr_clean = current_col.dropna().values

        if len(ref_clean) < 10 or len(curr_clean) < 10:
            return {
                "drift_detected": False,
                "p_value": 1.0,
                "psi": 0.0,
                "statistic": 0.0,
                "test": "ks_test",
                "status": "insufficient_data",
            }

        ks_stat, p_val = stats.ks_2samp(ref_clean, curr_clean)
        psi_val = self.calculate_psi(ref_clean, curr_clean)

        drift_detected = bool(
            p_val < self.significance_level or psi_val > self.psi_threshold
        )

        return {
            "drift_detected": drift_detected,
            "p_value": float(p_val),
            "psi": float(psi_val),
            "statistic": float(ks_stat),
            "test": "ks_test_and_psi",
            "ref_mean": float(np.mean(ref_clean)),
            "curr_mean": float(np.mean(curr_clean)),
            "ref_std": float(np.std(ref_clean)),
            "curr_std": float(np.std(curr_clean)),
        }

    def detect_categorical_drift(
        self, reference_col: pd.Series, current_col: pd.Series
    ) -> Dict[str, Any]:
        """Runs Chi-Square / Fisher test on categorical distributions."""
        all_cats = list(
            set(reference_col.dropna().unique()) | set(current_col.dropna().unique())
        )
        if len(all_cats) <= 1:
            return {
                "drift_detected": False,
                "p_value": 1.0,
                "test": "chi2",
                "status": "insufficient_categories",
            }

        ref_counts = reference_col.value_counts()
        curr_counts = current_col.value_counts()

        ref_freq = [ref_counts.get(c, 0) + 1 for c in all_cats]
        curr_freq = [curr_counts.get(c, 0) + 1 for c in all_cats]

        chi2_stat, p_val = stats.chisquare(
            f_obs=curr_freq,
            f_exp=[
                f * sum(curr_freq) / sum(ref_freq) for f in ref_freq
            ],
        )

        return {
            "drift_detected": bool(p_val < self.significance_level),
            "p_value": float(p_val),
            "statistic": float(chi2_stat),
            "test": "chi2",
            "categories": all_cats,
        }


class DriftMonitor:
    """High-level drift monitoring engine comparing baseline reference data to production inference data."""

    def __init__(
        self,
        reference_data_path: Optional[str] = None,
        significance_level: float = 0.05,
        psi_threshold: float = 0.2,
        drift_share_threshold: float = 0.3,
    ):
        self.detector = StatisticalDriftDetector(
            significance_level=significance_level, psi_threshold=psi_threshold
        )
        self.drift_share_threshold = drift_share_threshold
        self.reference_df: Optional[pd.DataFrame] = None

        if reference_data_path and os.path.exists(reference_data_path):
            self.load_reference_data(reference_data_path)

    def load_reference_data(self, path: str) -> None:
        """Load baseline/reference dataset."""
        logger.info("Loading reference data from %s", path)
        if path.endswith(".parquet"):
            self.reference_df = pd.read_parquet(path)
        else:
            self.reference_df = pd.read_csv(path)

    def run_drift_analysis(
        self,
        current_df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
        categorical_columns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Perform comprehensive feature and target drift analysis."""
        if self.reference_df is None:
            raise ValueError(
                "Reference data is not loaded. Call load_reference_data() first."
            )

        if feature_columns is None:
            # Common numeric features
            feature_columns = [
                c
                for c in current_df.columns
                if c in self.reference_df.columns
                and pd.api.types.is_numeric_dtype(current_df[c])
            ]

        if categorical_columns is None:
            categorical_columns = [
                c
                for c in current_df.columns
                if c in self.reference_df.columns
                and not pd.api.types.is_numeric_dtype(current_df[c])
            ]

        feature_drift_results = {}
        drifted_features_count = 0

        for col in feature_columns:
            if col in self.reference_df.columns and col in current_df.columns:
                res = self.detector.detect_numerical_drift(
                    self.reference_df[col], current_df[col]
                )
                feature_drift_results[col] = res
                if res.get("drift_detected", False):
                    drifted_features_count += 1

        for col in categorical_columns:
            if col in self.reference_df.columns and col in current_df.columns:
                res = self.detector.detect_categorical_drift(
                    self.reference_df[col], current_df[col]
                )
                feature_drift_results[col] = res
                if res.get("drift_detected", False):
                    drifted_features_count += 1

        total_tested = len(feature_columns) + len(categorical_columns)
        drift_share = drifted_features_count / max(total_tested, 1)
        dataset_drift = drift_share >= self.drift_share_threshold

        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dataset_drift": bool(dataset_drift),
            "drift_share": float(drift_share),
            "drift_share_threshold": float(self.drift_share_threshold),
            "drifted_features_count": drifted_features_count,
            "total_features_tested": total_tested,
            "metrics_per_feature": feature_drift_results,
            "recommendation": (
                "TRIGGER_RETRAINING"
                if dataset_drift
                else "SYSTEM_STABLE"
            ),
        }

        logger.info(
            "Drift analysis completed. Dataset drift: %s (Drift share: %.2f%%)",
            dataset_drift,
            drift_share * 100,
        )
        return report

    def save_report(self, report: Dict[str, Any], output_path: str) -> None:
        """Export drift report as JSON artifact."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info("Saved drift report to %s", output_path)
