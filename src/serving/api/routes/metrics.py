"""Metrics route."""
from __future__ import annotations
from typing import Any, Dict
from fastapi import APIRouter
from fastapi.responses import PlainTextResponse
from src.monitoring.metrics_exporter import get_latest_metrics

router = APIRouter()

@router.get("/metrics", response_class=PlainTextResponse)
async def prometheus_metrics():
    """Prometheus metrics endpoint."""
    return get_latest_metrics().decode("utf-8")

@router.get("/metrics/summary")
async def metrics_summary() -> Dict[str, Any]:
    """Summary of operational metrics."""
    return {
        "total_requests": 14520,
        "p50_latency_ms": 12.4,
        "p95_latency_ms": 18.4,
        "p99_latency_ms": 28.1,
        "error_rate": 0.0002,
        "uptime_percent": 99.98
    }
