"""
Standard Library Unit Test Suite Runner
========================================
Runs unit tests for Risk Scorer, Drift Detector, and Quality Gate
without requiring external test runner dependencies.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.models.risk_scorer import RiskScorer
from src.monitoring.drift_detector import StatisticalDriftDetector
from src.training.quality_gate import ModelQualityGate


class TestRiskScorer(unittest.TestCase):
    def setUp(self):
        self.scorer = RiskScorer()

    def test_critical_zero_day_risk(self):
        result = self.scorer.score(
            cve_id="CVE-2024-9999",
            predicted_severity=4,  # CRITICAL
            model_confidence=0.98,
            in_cisa_kev=True,
            epss_score=0.98,
            asset_criticality=0.9,
            exposure_score=1.0,
        )
        self.assertIn(result.risk_tier, ["CRITICAL_RISK", "HIGH_RISK"])
        self.assertGreaterEqual(result.risk_score, 80.0)
        self.assertLessEqual(result.risk_score, 100.0)

    def test_score_boundaries(self):
        max_res = self.scorer.score(
            cve_id="CVE-MAX",
            predicted_severity=4,
            model_confidence=1.0,
            in_cisa_kev=True,
            epss_score=1.0,
            asset_criticality=1.0,
            exposure_score=1.0,
        )
        self.assertLessEqual(max_res.risk_score, 100.0)


class TestModelQualityGate(unittest.TestCase):
    def setUp(self):
        self.gate = ModelQualityGate(thresholds={
            "min_macro_f1": 0.80,
            "min_critical_recall": 0.85,
            "max_ece": 0.10,
            "max_latency_p99_ms": 50.0,
            "champion_improvement_margin": 0.0,
        })

    def test_quality_gate_pass(self):
        metrics = {
            "macro_f1": 0.88,
            "critical_recall": 0.92,
            "ece": 0.04,
            "latency_p99_ms": 22.0,
            "n_test_samples": 500,
        }
        report = self.gate.evaluate(metrics)
        self.assertTrue(report["passed"])

    def test_quality_gate_fail_critical_recall(self):
        metrics = {
            "macro_f1": 0.86,
            "critical_recall": 0.70,  # Below threshold
            "ece": 0.05,
            "latency_p99_ms": 20.0,
            "n_test_samples": 500,
        }
        report = self.gate.evaluate(metrics)
        self.assertFalse(report["passed"])


if __name__ == "__main__":
    unittest.main()
