"""Risk scoring route."""
from __future__ import annotations
import numpy as np
from fastapi import APIRouter, HTTPException, Request
from src.serving.api.schemas import RiskScoreRequest, RiskScoreResponse, RiskComponentsResponse
from src.serving.api.routes.predict import _build_feature_row, SEVERITY_LABELS
from src.models.risk_scorer import RiskScorer
import os

router = APIRouter()
_scorer = RiskScorer()


@router.post("/risk-score", response_model=RiskScoreResponse)
async def compute_risk_score(req: RiskScoreRequest, request: Request):
    """
    Compute composite risk prioritization score (0–100).

    Combines ML severity prediction, EPSS exploitation probability,
    CISA KEV status, asset criticality, and exposure into a single score.
    """
    registry = request.app.state.model_registry

    try:
        from src.serving.api.schemas import PredictRequest
        pred_req = PredictRequest(
            cve_id=req.cve_id,
            cvss_vector_string=req.cvss_vector_string,
            description=req.description,
            epss_score=req.epss_score,
            epss_percentile=req.epss_percentile,
            in_cisa_kev=req.in_cisa_kev,
            year_published=req.year_published,
            cwe_ids=req.cwe_ids,
            reference_count=req.reference_count,
            vendor_count=req.vendor_count,
        )
        X = _build_feature_row(pred_req)
        if registry.champion_loaded:
            proba = registry.predict(X)
        else:
            if req.in_cisa_kev or req.epss_score > 0.8:
                proba = np.array([[0.01, 0.02, 0.07, 0.20, 0.70]])
            elif req.epss_score > 0.4:
                proba = np.array([[0.02, 0.08, 0.20, 0.60, 0.10]])
            else:
                proba = np.array([[0.05, 0.25, 0.50, 0.15, 0.05]])

        pred_class = int(np.argmax(proba[0]))
        confidence = float(proba[0][pred_class])

        result = _scorer.score(
            cve_id=req.cve_id,
            predicted_severity=pred_class,
            model_confidence=confidence,
            in_cisa_kev=req.in_cisa_kev,
            epss_score=req.epss_score,
            asset_criticality=req.asset_criticality,
            exposure_score=req.exposure_score,
        )

        return RiskScoreResponse(
            cve_id=req.cve_id,
            risk_score=result.risk_score,
            risk_tier=result.risk_tier,
            predicted_severity=SEVERITY_LABELS[pred_class],
            model_confidence=round(confidence, 4),
            components=RiskComponentsResponse(
                cvss_component=result.components.cvss_component,
                kev_component=result.components.kev_component,
                epss_component=result.components.epss_component,
                asset_criticality=result.components.asset_criticality,
                exposure_score=result.components.exposure_score,
                confidence_component=result.components.confidence_component,
            ),
            component_contributions=result.component_contributions,
            weights_used=result.weights_used,
            model_version=registry.champion_version,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
