"""Analyst feedback route."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional
from fastapi import APIRouter
from pydantic import BaseModel, Field

from src.serving.api.db import log_feedback_to_db

router = APIRouter()


class FeedbackPayload(BaseModel):
    cve_id: str
    corrected_severity: str
    analyst_id: str
    predicted_severity: Optional[str] = None
    notes: str = ""
    confidence_rating: int = Field(5, ge=1, le=5)


@router.post("/feedback")
async def submit_feedback(payload: FeedbackPayload) -> Dict[str, Any]:
    """
    Record human-in-the-loop analyst feedback into the analyst_feedback table
    in PostgreSQL for model retraining datasets.
    """
    saved_to_db = log_feedback_to_db(
        cve_id=payload.cve_id,
        analyst_id=payload.analyst_id,
        predicted_severity=payload.predicted_severity,
        corrected_severity=payload.corrected_severity.upper(),
        feedback_notes=payload.notes,
        confidence_rating=payload.confidence_rating,
    )

    return {
        "status": "success",
        "message": f"Feedback for {payload.cve_id} logged successfully.",
        "stored_in_db": saved_to_db,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "recorded": payload.dict(),
    }
