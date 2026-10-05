"""
Prometheus Metrics Exporter
===========================
Exports Prometheus counter, histogram, and gauge metrics for real-time observability.
"""

from __future__ import annotations

import logging
from typing import Optional
from prometheus_client import Counter, Gauge, Histogram, generate_latest

logger = logging.getLogger("mlops.monitoring.prometheus")

# Prometheus Metric Definitions
PREDICTION_REQUESTS = Counter(
    "vulnerability_prediction_requests_total",
    "Total number of vulnerability severity prediction requests",
    ["model_version", "status"],
)

PREDICTION_LATENCY = Histogram(
    "vulnerability_prediction_duration_seconds",
    "Latency of vulnerability severity predictions",
    ["model_version"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)

SEVERITY_PREDICTIONS = Counter(
    "vulnerability_severity_predictions_total",
    "Predictions broken down by predicted severity tier",
    ["severity_tier"],
)

RISK_SCORE_GAUGE = Histogram(
    "vulnerability_risk_score_distribution",
    "Distribution of calculated composite risk scores",
    buckets=[20, 40, 60, 75, 90, 100],
)

ACTIVE_CHAMPION_VERSION = Gauge(
    "vulnerability_champion_model_version",
    "Current champion model version registered in MLflow",
)

DATA_DRIFT_SCORE = Gauge(
    "vulnerability_data_drift_share",
    "Current dataset drift share percentage",
)


def get_latest_metrics() -> bytes:
    """Return raw Prometheus exposition format text."""
    return generate_latest()
