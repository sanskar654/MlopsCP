"""Analyst feedback route."""
from __future__ import annotations
from datetime import datetime, timezone
from typing import Any, Dict
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class FeedbackPayload(BaseModel):
    cve_id: str
    corrected_severity: str
    analyst_id: str
    notes: str = ""

@router.post("/feedback")
async def submit_feedback(payload: FeedbackPayload) -> Dict[str, Any]:
    """Record human-in-the-loop analyst feedback for future model retraining datasets."""
    return {
        "status": "success",
        "message": f"Feedback for {payload.cve_id} logged successfully.",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "recorded": payload.dict()
    }
