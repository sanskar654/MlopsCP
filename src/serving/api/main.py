"""
FastAPI Application Entry Point
================================
Production-grade ML inference API for vulnerability severity prediction
and risk prioritization.

Endpoints:
  POST /api/v1/predict          - Severity prediction with confidence
  POST /api/v1/risk-score       - Full composite risk score
  POST /api/v1/explain          - SHAP-based explanation
  GET  /api/v1/vulnerabilities  - Paginated vulnerability list
  GET  /api/v1/vulnerabilities/{cve_id} - Single CVE details
  POST /api/v1/feedback         - Analyst label correction
  GET  /api/v1/models           - Model registry info
  GET  /api/v1/models/compare   - Champion vs challenger metrics
  GET  /api/v1/drift            - Latest drift report
  GET  /api/v1/metrics          - Model performance metrics
  POST /api/v1/retrain          - Trigger retraining (admin)
  GET  /health                  - Health check
  GET  /metrics                 - Prometheus metrics
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

try:
    import mlflow
    import mlflow.sklearn
    HAS_MLFLOW = True
except ImportError:
    mlflow = None
    HAS_MLFLOW = False

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

try:
    from loguru import logger
except ImportError:
    logger = logging.getLogger("serving.api")

from .model_loader import ModelRegistry
from .routes import (
    drift,
    explain,
    feedback,
    health,
    metrics,
    models,
    predict,
    retrain,
    risk,
    vulnerabilities,
)

# ─── Logging Setup ──────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)

# ─── Application State ──────────────────────────────────────────────────────
model_registry = ModelRegistry()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle management."""
    logger.info("🚀 Starting Vulnerability MLOps API...")

    # Configure MLflow
    if HAS_MLFLOW and mlflow is not None:
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        try:
            mlflow.set_tracking_uri(mlflow_uri)
            logger.info(f"MLflow tracking: {mlflow_uri}")
        except Exception as e:
            logger.warning(f"Could not connect to MLflow: {e}")

    # Load champion model
    champion_name = os.getenv(
        "CHAMPION_MODEL_NAME", "vulnerability-severity-classifier"
    )
    try:
        model_registry.load_champion(champion_name)
        logger.info(f"✅ Champion model loaded: {champion_name}")
    except Exception as e:
        logger.warning(f"⚠️  Champion model not available: {e}. Running in demo mode.")

    # Load challenger model (optional, for shadow mode)
    if os.getenv("ENABLE_SHADOW_MODE", "false").lower() == "true":
        challenger_name = os.getenv(
            "CHALLENGER_MODEL_NAME", "vulnerability-severity-challenger"
        )
        try:
            model_registry.load_challenger(challenger_name)
            logger.info(f"✅ Challenger model loaded: {challenger_name}")
        except Exception as e:
            logger.debug(f"No challenger model: {e}")

    # Expose model_registry to routes
    app.state.model_registry = model_registry
    logger.info("✅ API startup complete")

    yield

    logger.info("Shutting down API...")


# ─── FastAPI App ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="Vulnerability MLOps API",
    description=(
        "Adaptive MLOps platform for software vulnerability detection and "
        "risk prioritization. Powered by XGBoost + SHAP + MLflow."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

@app.get("/api/docs", include_in_schema=False)
async def redirect_api_docs():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")

app.state.model_registry = model_registry

# ─── Middleware ───────────────────────────────────────────────────────────────
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv(
        "ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8080"
    ).split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests with timing."""
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        f"{request.method} {request.url.path} "
        f"→ {response.status_code} ({duration_ms:.1f}ms)"
    )
    response.headers["X-Response-Time-Ms"] = f"{duration_ms:.1f}"
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "path": str(request.url.path)},
    )


# ─── Route Registration ───────────────────────────────────────────────────────
app.include_router(health.router, tags=["Health"])
app.include_router(predict.router, prefix="/api/v1", tags=["Prediction"])
app.include_router(risk.router, prefix="/api/v1", tags=["Risk Scoring"])
app.include_router(explain.router, prefix="/api/v1", tags=["Explainability"])
app.include_router(vulnerabilities.router, prefix="/api/v1", tags=["Vulnerabilities"])
app.include_router(feedback.router, prefix="/api/v1", tags=["Feedback"])
app.include_router(models.router, prefix="/api/v1", tags=["Model Registry"])
app.include_router(drift.router, prefix="/api/v1", tags=["Drift Monitoring"])
app.include_router(metrics.router, prefix="/api/v1", tags=["Metrics"])
app.include_router(retrain.router, prefix="/api/v1", tags=["Operations"])


from pathlib import Path
from fastapi.staticfiles import StaticFiles

dashboard_path = Path(__file__).resolve().parent.parent.parent / "dashboard"
if dashboard_path.exists():
    app.mount("/", StaticFiles(directory=str(dashboard_path), html=True), name="dashboard")
else:
    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "service": "Vulnerability MLOps API",
            "version": "1.0.0",
            "docs": "/api/docs",
            "health": "/health",
        }
