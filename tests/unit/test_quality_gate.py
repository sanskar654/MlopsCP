"""
Unit Tests: Model Quality Gate
==============================
Verifies challenger rejection/promotion logic against thresholds.
"""

import pytest
from training.quality_gate import ModelQualityGate


@pytest.fixture
def quality_gate():
    return ModelQualityGate(thresholds={
        "min_macro_f1": 0.80,
        "min_critical_recall": 0.85,
        "max_ece": 0.10,
        "max_latency_p99_ms": 50.0,
        "champion_improvement_margin": 0.0,
    })


def test_quality_gate_pass(quality_gate):
    metrics = {
        "macro_f1": 0.88,
        "critical_recall": 0.92,
        "ece": 0.04,
        "latency_p99_ms": 22.0,
        "n_test_samples": 500,
    }
    report = quality_gate.evaluate(metrics)
    assert report["passed"] is True
    assert "checks" in report


def test_quality_gate_fail_critical_recall(quality_gate):
    metrics = {
        "macro_f1": 0.86,
        "critical_recall": 0.70,  # Below 0.85 threshold!
        "ece": 0.05,
        "latency_p99_ms": 20.0,
        "n_test_samples": 500,
    }
    report = quality_gate.evaluate(metrics)
    assert report["passed"] is False
