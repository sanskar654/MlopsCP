"""
Unit Tests: Risk Scorer
=======================
Verifies weighted formula, boundary limits, and tier classifications.
"""

import pytest
from models.risk_scorer import RiskScorer


@pytest.fixture
def risk_scorer():
    return RiskScorer()


def test_critical_zero_day_risk(risk_scorer):
    result = risk_scorer.score(
        cve_id="CVE-2024-9999",
        predicted_severity=4,  # CRITICAL
        model_confidence=0.98,
        in_cisa_kev=True,
        epss_score=0.98,
        asset_criticality=0.9,
        exposure_score=1.0,
    )
    assert result.risk_tier in ["CRITICAL_RISK", "HIGH_RISK"]
    assert result.risk_score >= 80.0
    assert result.risk_score <= 100.0


def test_low_internal_risk(risk_scorer):
    result = risk_scorer.score(
        cve_id="CVE-2024-0001",
        predicted_severity=1,  # LOW
        model_confidence=0.90,
        in_cisa_kev=False,
        epss_score=0.01,
        asset_criticality=0.1,
        exposure_score=0.1,
    )
    assert result.risk_tier in ["LOW_RISK", "MEDIUM_RISK"]
    assert result.risk_score <= 40.0


def test_score_boundaries(risk_scorer):
    # Test maximum possible inputs
    max_res = risk_scorer.score(
        cve_id="CVE-MAX",
        predicted_severity=4,
        model_confidence=1.0,
        in_cisa_kev=True,
        epss_score=1.0,
        asset_criticality=1.0,
        exposure_score=1.0,
    )
    assert max_res.risk_score <= 100.0

    # Test minimum possible inputs
    min_res = risk_scorer.score(
        cve_id="CVE-MIN",
        predicted_severity=0,
        model_confidence=0.0,
        in_cisa_kev=False,
        epss_score=0.0,
        asset_criticality=0.0,
        exposure_score=0.0,
    )
    assert min_res.risk_score >= 0.0
