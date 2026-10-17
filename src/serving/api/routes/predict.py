"""
Prediction Route
================
POST /api/v1/predict — Symptom-based severity prediction with calibrated confidence.
Logs every request and prediction to `predictions_log` table in PostgreSQL.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Request

try:
    from loguru import logger
except ImportError:
    logger = logging.getLogger("serving.api.predict")

from src.features.feature_engineer import (
    ALL_FEATURES,
    FeatureEngineer,
)
from src.serving.api.db import log_prediction_to_db
from src.serving.api.schemas import PredictRequest, PredictResponse

router = APIRouter()

SEVERITY_LABELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
_engineer = FeatureEngineer()


def _build_feature_row(req: PredictRequest) -> pd.DataFrame:
    """
    Build symptom-based feature vector DataFrame from request.
    Strictly NO CVSS vector submetrics (AV, AC, PR, UI, S, C, I, A).
    Strictly NO post-publication signals (EPSS, KEV, CVSS score).
    """
    raw_dict = {
        "cve_id": req.cve_id,
        "description": req.description or "",
        "cwe_ids": req.cwe_ids,
        "vendor_count": req.vendor_count,
        "reference_count": req.reference_count,
        "year_published": req.year_published or 2024,
    }
    df_raw = pd.DataFrame([raw_dict])
    df_features = _engineer.transform(df_raw)
    cols_to_use = [c for c in ALL_FEATURES if c in df_features.columns]
    return df_features[cols_to_use].fillna(0)


@router.post("/predict", response_model=PredictResponse)
async def predict_severity(req: PredictRequest, request: Request):
    """
    Predict vulnerability severity strictly from pre-publication symptoms:
    description text (TF-IDF & security keywords), CWE category, vendor/product,
    and reference metadata.

    Logs inference event to `predictions_log` in PostgreSQL.
    """
    registry = request.app.state.model_registry

    try:
        X = _build_feature_row(req)
        model_version = registry.champion_version if registry else "v2.0-baseline"

        if registry and registry.champion_loaded:
            proba = registry.predict(X)  # shape [1, N_CLASSES]
        else:
            # Heuristic symptom-based fallback for cold start / offline demo
            desc_lower = (req.description or "").lower()
            if any(k in desc_lower for k in ["remote code", "rce", "arbitrary code", "buffer overflow"]):
                proba = np.array([[0.02, 0.08, 0.25, 0.65]])
            elif any(k in desc_lower for k in ["sql inject", "privilege escal", "unauthenticated"]):
                proba = np.array([[0.05, 0.15, 0.65, 0.15]])
            elif any(k in desc_lower for k in ["cross-site script", "xss", "traversal"]):
                proba = np.array([[0.10, 0.65, 0.20, 0.05]])
            else:
                proba = np.array([[0.40, 0.40, 0.15, 0.05]])

        pred_class = int(np.argmax(proba[0]))
        num_classes = proba.shape[1]
        labels = SEVERITY_LABELS if num_classes == 4 else ["NONE", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
        pred_label = labels[pred_class] if pred_class < len(labels) else "HIGH"
        confidence = float(proba[0][pred_class])

        probabilities = {
            labels[i]: round(float(proba[0][i]), 4)
            for i in range(min(num_classes, len(labels)))
        }

        # ─── Log Prediction to PostgreSQL (MLOps Requirement 1) ─────────────────
        features_snapshot = {
            "description_length": len(req.description or ""),
            "cwe_ids": req.cwe_ids,
            "vendor_count": req.vendor_count,
            "reference_count": req.reference_count,
            "year_published": req.year_published,
        }
        log_prediction_to_db(
            cve_id=req.cve_id,
            model_version=model_version,
            prediction=pred_label,
            confidence=confidence,
            features_snapshot=features_snapshot,
        )

        # Shadow mode: evaluate challenger if active
        if registry and registry.challenger_loaded:
            try:
                challenger_proba = registry.predict_challenger(X)
                if challenger_proba is not None:
                    challenger_class = int(np.argmax(challenger_proba[0]))
                    if challenger_class != pred_class:
                        logger.debug(
                            f"Shadow disagreement on {req.cve_id}: "
                            f"champion={pred_label}, challenger={labels[challenger_class]}"
                        )
            except Exception:
                pass

        return PredictResponse(
            cve_id=req.cve_id,
            predicted_severity=pred_label,
            predicted_class=pred_class,
            confidence=round(confidence, 4),
            probabilities=probabilities,
            model_version=model_version,
            prediction_id=str(uuid.uuid4()),
        )

    except Exception as e:
        logger.error(f"Prediction failed for {req.cve_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
