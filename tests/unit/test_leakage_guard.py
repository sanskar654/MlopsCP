"""
Unit Test: Data Leakage Guard
==============================
Verifies that Stage 1 Feature Engineering Matrix (X) excludes target labels
(cvss_score, cvss_severity) and Stage 2 threat components (epss_score, kev_flag).
"""

import pandas as pd
import pytest
from src.features.feature_engineer import FeatureEngineer


@pytest.fixture
def sample_raw_vulnerabilities():
    return pd.DataFrame([
        {
            "cve_id": "CVE-2024-1001",
            "published_date": "2024-01-15",
            "cvss_vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
            "cvss_score": 9.8,
            "cvss_severity": "CRITICAL",
            "cwe": "CWE-79",
            "description": "Remote code execution via SQL injection vulnerability",
            "vendor": "apache",
            "product": "httpd",
            "epss_score": 0.95,
            "in_cisa_kev": True,
        },
        {
            "cve_id": "CVE-2024-1002",
            "published_date": "2024-02-10",
            "cvss_vector": "CVSS:3.1/AV:L/AC:H/PR:L/UI:R/S:U/C:L/I:N/A:N",
            "cvss_score": 2.5,
            "cvss_severity": "LOW",
            "cwe": "CWE-200",
            "description": "Information disclosure in local log output",
            "vendor": "linux",
            "product": "kernel",
            "epss_score": 0.01,
            "in_cisa_kev": False,
        }
    ])


def test_stage1_feature_matrix_leakage_guard(sample_raw_vulnerabilities):
    engineer = FeatureEngineer()
    X = engineer.fit_transform(sample_raw_vulnerabilities)

    forbidden_target_columns = [
        "cvss_score",
        "cvss_severity",
        "severity",
        "target",
        "label",
    ]

    forbidden_stage2_columns = [
        "epss_score",
        "epss_percentile",
        "epss",
        "in_cisa_kev",
        "kev_flag",
        "cisa_kev_active",
    ]

    for col in forbidden_target_columns:
        assert col not in X.columns, f"Target Leak Detected! Column '{col}' found in Stage 1 features X"

    for col in forbidden_stage2_columns:
        assert col not in X.columns, f"Stage 2 Leak Detected! Column '{col}' found in Stage 1 features X"
