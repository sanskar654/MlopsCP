"""Health check route."""
from __future__ import annotations
import os
from datetime import datetime, timezone
from fastapi import APIRouter, Request
from src.serving.api.schemas import HealthResponse

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request):
    registry = getattr(request.app.state, "model_registry", None)
    db_status = "unknown"
    mlflow_status = "unknown"
    try:
        import mlflow
        mlflow.get_tracking_uri()
        mlflow_status = "ok"
    except Exception:
        mlflow_status = "error"
    try:
        import sqlalchemy
        from sqlalchemy import create_engine, text
        engine = create_engine(os.getenv("DATABASE_URL", ""))
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_status = "ok"
    except Exception:
        db_status = "error"

    return HealthResponse(
        status="healthy" if (registry and registry.champion_loaded) else "degraded",
        version="1.0.0",
        model_loaded=registry.champion_loaded if registry else False,
        champion_model=registry.champion_name if registry else None,
        challenger_model=registry.challenger_name if (registry and registry.challenger_loaded) else None,
        database=db_status,
        mlflow=mlflow_status,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
