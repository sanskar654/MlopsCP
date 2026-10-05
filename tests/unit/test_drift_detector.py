"""
Unit Tests: Drift Detector
==========================
Verifies statistical testing (KS-test, PSI) on identical vs shifted distributions.
"""

import numpy as np
import pandas as pd
import pytest
from monitoring.drift_detector import StatisticalDriftDetector, DriftMonitor


def test_psi_identical_distribution():
    data = np.random.normal(0, 1, 1000)
    psi = StatisticalDriftDetector.calculate_psi(data, data)
    assert psi < 0.05  # Should be near 0 for identical distribution


def test_psi_shifted_distribution():
    ref = np.random.normal(0, 1, 1000)
    curr = np.random.normal(3, 1, 1000)  # Significant mean shift
    psi = StatisticalDriftDetector.calculate_psi(ref, curr)
    assert psi > 0.25  # Should exceed PSI warning threshold


def test_drift_monitor_analysis():
    ref_df = pd.DataFrame({
        "epss_score": np.random.uniform(0.01, 0.2, 500),
        "feature_a": np.random.normal(10, 2, 500),
    })
    curr_df = pd.DataFrame({
        "epss_score": np.random.uniform(0.7, 0.99, 500),  # Drifted
        "feature_a": np.random.normal(10, 2, 500),       # Stable
    })

    monitor = DriftMonitor()
    monitor.reference_df = ref_df
    report = monitor.run_drift_analysis(curr_df, feature_columns=["epss_score", "feature_a"])

    assert "epss_score" in report["metrics_per_feature"]
    assert report["metrics_per_feature"]["epss_score"]["drift_detected"] is True
