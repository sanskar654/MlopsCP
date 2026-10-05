"""
Prediction Route
================
POST /api/v1/predict  — Severity prediction with calibrated confidence.
"""

from __future__ import annotations

import uuid
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Request
import logging

try:
    from loguru import logger
except ImportError:
    logger = logging.getLogger("serving.api.predict")

from src.features.cvss_parser import parse_cvss_vector, cvss_submetrics_to_onehot
from src.features.feature_engineer import ALL_FEATURES
from src.serving.api.schemas import PredictRequest, PredictResponse

router = APIRouter()

SEVERITY_LABELS = ["NONE", "LOW", "MEDIUM", "HIGH", "CRITICAL"]


def _build_feature_row(req: PredictRequest) -> pd.DataFrame:
    """Convert API request to a feature vector DataFrame."""
    features: dict[str, Any] = {f: 0 for f in ALL_FEATURES}

    # CVSS submetric one-hot
    parsed = parse_cvss_vector(req.cvss_vector_string)
    onehot = cvss_submetrics_to_onehot(parsed)
    features.update(onehot)

    # Exploitation signals
    features["in_cisa_kev"] = int(req.in_cisa_kev)
    features["epss_score"] = req.epss_score
    features["epss_percentile"] = req.epss_percentile

    # Temporal
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    if req.year_published:
        features["year_published"] = req.year_published
        features["days_since_published"] = (now.year - req.year_published) * 365
        features["quarter_published"] = 1
        features["is_recent"] = int(features["days_since_published"] < 90)

    # Text features from description
    if req.description:
        import re
        desc = req.description
        features["description_length"] = len(desc)
        features["description_word_count"] = len(desc.split())
        features["has_rce_keyword"] = int(bool(re.search(
            r"\b(remote.?code.?exec|rce|arbitrary.?code)\b", desc, re.I)))
        features["has_sqli_keyword"] = int(bool(re.search(
            r"\b(sql.?inject|sqli)\b", desc, re.I)))
        features["has_xss_keyword"] = int(bool(re.search(
            r"\b(cross.?site.?script|xss)\b", desc, re.I)))
        features["has_overflow_keyword"] = int(bool(re.search(
            r"\b(buffer.?overflow|stack.?overflow|memory.?corrupt)\b", desc, re.I)))
        features["has_auth_bypass_keyword"] = int(bool(re.search(
            r"\b(auth.?bypass|unauthenticated)\b", desc, re.I)))
        features["has_priv_esc_keyword"] = int(bool(re.search(
            r"\b(privilege.?escal|priv.?esc)\b", desc, re.I)))
        features["has_dos_keyword"] = int(bool(re.search(
            r"\b(denial.?of.?service|dos|crash)\b", desc, re.I)))

    # CWE features
    from features.feature_engineer import (
        MEMORY_SAFETY_CWES, INJECTION_CWES, AUTH_CWES, CONFIG_CWES
    )
    cwe_set = set(req.cwe_ids)
    features["is_memory_safety_cwe"] = int(bool(cwe_set & MEMORY_SAFETY_CWES))
    features["is_injection_cwe"] = int(bool(cwe_set & INJECTION_CWES))
    features["is_auth_cwe"] = int(bool(cwe_set & AUTH_CWES))
    features["is_config_cwe"] = int(bool(cwe_set & CONFIG_CWES))

    # Metadata
    features["reference_count"] = req.reference_count
    features["vendor_count"] = req.vendor_count

    # Ensure all feature columns exist and are in correct order
    row_data = {col: features.get(col, 0) for col in ALL_FEATURES}
    return pd.DataFrame([row_data])


@router.post("/predict", response_model=PredictResponse)
async def predict_severity(req: PredictRequest, request: Request):
    """
    Predict vulnerability severity with calibrated confidence scores.

    Returns predicted severity label (NONE/LOW/MEDIUM/HIGH/CRITICAL),
    class probabilities, and model confidence.
    """
    registry = request.app.state.model_registry

    try:
        X = _build_feature_row(req)
        if registry.champion_loaded:
            proba = registry.predict(X)  # shape [1, 5]
        else:
            # Heuristic CVSS fallback for cold start / offline demo
            if req.in_cisa_kev or req.epss_score > 0.8:
                proba = np.array([[0.01, 0.02, 0.07, 0.20, 0.70]])
            elif req.epss_score > 0.4:
                proba = np.array([[0.02, 0.08, 0.20, 0.60, 0.10]])
            else:
                proba = np.array([[0.05, 0.25, 0.50, 0.15, 0.05]])

        pred_class = int(np.argmax(proba[0]))
        confidence = float(proba[0][pred_class])

        # Shadow mode: log challenger prediction asynchronously
        if registry.challenger_loaded:
            try:
                challenger_proba = registry.predict_challenger(X)
                if challenger_proba is not None:
                    challenger_class = int(np.argmax(challenger_proba[0]))
                    if challenger_class != pred_class:
                        logger.debug(
                            f"Shadow disagreement on {req.cve_id}: "
                            f"champion={SEVERITY_LABELS[pred_class]}, "
                            f"challenger={SEVERITY_LABELS[challenger_class]}"
                        )
            except Exception:
                pass

        return PredictResponse(
            cve_id=req.cve_id,
            predicted_severity=SEVERITY_LABELS[pred_class],
            predicted_class=pred_class,
            confidence=round(confidence, 4),
            probabilities={
                SEVERITY_LABELS[i]: round(float(proba[0][i]), 4)
                for i in range(5)
            },
            model_version=registry.champion_version,
            prediction_id=str(uuid.uuid4()),
        )

    except Exception as e:
        logger.error(f"Prediction failed for {req.cve_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
