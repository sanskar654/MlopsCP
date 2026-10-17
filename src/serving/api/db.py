"""
Database Client & Storage Helpers
=================================
Manages PostgreSQL connections with connection pooling, graceful fallback,
and helper functions for:
- Logging predictions to `predictions_log`
- Logging human analyst feedback to `analyst_feedback`
- Querying recent production predictions for drift detection
- Fetching unretrained feedback records for model retraining
- Updating late ground-truth labels from NVD
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger("serving.database")

_engine: Optional[Engine] = None
_last_failure_time: float = 0.0
FAILURE_COOLDOWN_SECONDS: float = 30.0


def get_db_engine() -> Optional[Engine]:
    """Get or initialize the SQLAlchemy engine with failure cooldown."""
    global _engine, _last_failure_time
    if _engine is not None:
        return _engine

    now = time.time()
    if now - _last_failure_time < FAILURE_COOLDOWN_SECONDS:
        return None

    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        postgres_user = os.getenv("POSTGRES_USER", "mlops_user")
        postgres_password = os.getenv(
            "POSTGRES_PASSWORD", "mlops_secure_pass_change_me"
        )
        postgres_host = os.getenv("POSTGRES_HOST", "postgres")
        postgres_port = os.getenv("POSTGRES_PORT", "5432")
        postgres_db = os.getenv("POSTGRES_DB", "vuln_mlops")
        db_url = f"postgresql://{postgres_user}:{postgres_password}@{postgres_host}:{postgres_port}/{postgres_db}"

    try:
        _engine = create_engine(
            db_url,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            connect_args={"connect_timeout": 1},
        )
        with _engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("Successfully connected to PostgreSQL database.")
        return _engine
    except Exception as e:
        logger.warning(
            f"Database connection unavailable ({e}). Running in fallback mode."
        )
        _engine = None
        _last_failure_time = now
        return None


def log_prediction_to_db(
    cve_id: str,
    model_version: str,
    prediction: str,
    confidence: float,
    features_snapshot: Dict[str, Any],
) -> bool:
    """
    Save each inference request to predictions_log table in Postgres.
    """
    engine = get_db_engine()
    if engine is None:
        logger.debug(
            f"[Fallback] Prediction logged locally: cve_id={cve_id}, pred={prediction}"
        )
        return False

    try:
        features_json = json.dumps(features_snapshot)
        with engine.begin() as conn:
            query = text(
                """
                INSERT INTO predictions_log (cve_id, model_version, predicted_at, prediction, confidence, features_snapshot)
                VALUES (:cve_id, :model_version, :predicted_at, :prediction, :confidence, CAST(:features_snapshot AS JSONB))
            """
            )
            conn.execute(
                query,
                {
                    "cve_id": cve_id,
                    "model_version": model_version,
                    "predicted_at": datetime.now(timezone.utc),
                    "prediction": prediction,
                    "confidence": round(float(confidence), 4),
                    "features_snapshot": features_json,
                },
            )
        logger.debug(f"Logged prediction to DB for {cve_id} (prediction={prediction})")
        return True
    except SQLAlchemyError as e:
        logger.warning(f"Failed to log prediction to PostgreSQL for {cve_id}: {e}")
        return False


def log_feedback_to_db(
    cve_id: str,
    analyst_id: str,
    predicted_severity: Optional[str],
    corrected_severity: str,
    feedback_notes: str = "",
    confidence_rating: int = 5,
) -> bool:
    """
    Save analyst feedback into analyst_feedback table in Postgres.
    """
    engine = get_db_engine()
    if engine is None:
        logger.info(
            f"[Fallback] Feedback logged locally: cve_id={cve_id}, corrected={corrected_severity}"
        )
        return False

    try:
        with engine.begin() as conn:
            query = text(
                """
                INSERT INTO analyst_feedback (cve_id, analyst_id, predicted_severity, corrected_severity, feedback_notes, confidence_rating, used_for_retraining, submitted_at)
                VALUES (:cve_id, :analyst_id, :predicted_severity, :corrected_severity, :feedback_notes, :confidence_rating, FALSE, :submitted_at)
            """
            )
            conn.execute(
                query,
                {
                    "cve_id": cve_id,
                    "analyst_id": analyst_id,
                    "predicted_severity": predicted_severity,
                    "corrected_severity": corrected_severity,
                    "feedback_notes": feedback_notes,
                    "confidence_rating": confidence_rating,
                    "submitted_at": datetime.now(timezone.utc),
                },
            )
        logger.info(
            f"Recorded analyst feedback in DB for {cve_id}: corrected={corrected_severity}"
        )
        return True
    except SQLAlchemyError as e:
        logger.error(f"Failed to store feedback in PostgreSQL: {e}")
        return False


def get_recent_predictions(days: int = 7, limit: int = 5000) -> pd.DataFrame:
    """
    Fetch predictions from predictions_log over the last N days for drift checking.
    """
    engine = get_db_engine()
    if engine is None:
        logger.warning(
            "No DB connection for recent predictions. Returning empty DataFrame."
        )
        return pd.DataFrame()

    try:
        since = datetime.now(timezone.utc) - timedelta(days=days)
        query = text(
            """
            SELECT cve_id, model_version, predicted_at, prediction, confidence, features_snapshot
            FROM predictions_log
            WHERE predicted_at >= :since
            ORDER BY predicted_at DESC
            LIMIT :limit
        """
        )
        with engine.connect() as conn:
            df = pd.read_sql_query(query, conn, params={"since": since, "limit": limit})
        return df
    except SQLAlchemyError as e:
        logger.error(f"Failed to fetch recent predictions: {e}")
        return pd.DataFrame()


def get_unretrained_feedback() -> pd.DataFrame:
    """
    Retrieve analyst corrections not yet included in model retraining.
    """
    engine = get_db_engine()
    if engine is None:
        return pd.DataFrame()

    try:
        query = text(
            """
            SELECT id, cve_id, analyst_id, predicted_severity, corrected_severity, feedback_notes, confidence_rating, submitted_at
            FROM analyst_feedback
            WHERE used_for_retraining = FALSE
        """
        )
        with engine.connect() as conn:
            df = pd.read_sql_query(query, conn)
        return df
    except SQLAlchemyError as e:
        logger.error(f"Failed to fetch unretrained feedback: {e}")
        return pd.DataFrame()


def mark_feedback_used(feedback_ids: List[int]) -> bool:
    """
    Mark feedback records as incorporated into retraining.
    """
    if not feedback_ids:
        return True

    engine = get_db_engine()
    if engine is None:
        return False

    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    "UPDATE analyst_feedback SET used_for_retraining = TRUE WHERE id = ANY(:ids)"
                ),
                {"ids": feedback_ids},
            )
        return True
    except SQLAlchemyError as e:
        logger.error(f"Failed to mark feedback as used: {e}")
        return False
