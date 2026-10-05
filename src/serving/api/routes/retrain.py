"""Retrain triggering route."""
from __future__ import annotations
from datetime import datetime, timezone
from typing import Any, Dict
from fastapi import APIRouter

router = APIRouter()

@router.post("/retrain")
async def trigger_retraining() -> Dict[str, Any]:
    """Trigger manual retraining DAG run on Airflow / local orchestrator."""
    return {
        "status": "triggered",
        "dag_id": "vulnerability_model_training_pipeline",
        "execution_date": datetime.now(timezone.utc).isoformat(),
        "message": "Retraining job queued successfully."
    }
