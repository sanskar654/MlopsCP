"""
Integration Tests: FastAPI Serving Layer
========================================
Tests live request/response handling, input schema validation, and health checks.
"""

import pytest
from fastapi.testclient import TestClient
from serving.api.main import app

client = TestClient(app)


def test_health_check_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] in ["healthy", "degraded"]


def test_risk_scoring_endpoint():
    payload = {
        "cve_id": "CVE-2024-9999",
        "description": "Critical unauthenticated remote code execution vulnerability in core server",
        "cvss_vector_string": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
        "epss_score": 0.95,
        "in_cisa_kev": True,
        "asset_criticality": 0.95,
        "exposure_score": 0.95,
    }
    response = client.post("/api/v1/risk-score", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "risk_score" in data
    assert "risk_tier" in data
    assert data["risk_tier"] in ["CRITICAL_RISK", "HIGH_RISK"]
    assert data["risk_score"] >= 80.0


def test_predict_endpoint_validation():
    # Test valid input
    payload = {
        "cve_id": "CVE-2024-1234",
        "description": "Buffer overflow in protocol parser",
        "cvss_vector_string": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
        "epss_score": 0.5,
        "in_cisa_kev": False,
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_severity" in data
    assert "confidence" in data
