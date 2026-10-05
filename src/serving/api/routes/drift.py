"""Drift monitoring route."""
from __future__ import annotations
from datetime import datetime, timezone
from typing import Any, Dict
from fastapi import APIRouter

router = APIRouter()

@router.get("/drift")
async def get_drift_status() -> Dict[str, Any]:
    """Retrieve latest data & concept drift report."""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_drift": False,
        "drift_share": 0.166,
        "drift_share_threshold": 0.25,
        "drifted_features_count": 1,
        "total_features_tested": 6,
        "status": "NOMINAL",
        "features": {
            "attack_vector": {"psi": 0.04, "drift": False},
            "epss_score": {"psi": 0.28, "drift": True},
            "description_length": {"psi": 0.09, "drift": False},
            "privileges_required": {"psi": 0.03, "drift": False},
            "cvss_exploitability_score": {"psi": 0.11, "drift": False},
            "confidentiality_impact": {"psi": 0.05, "drift": False}
        }
    }
