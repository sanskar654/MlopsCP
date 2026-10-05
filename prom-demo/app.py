"""
Prometheus demo app for the Vulnerability Prioritization project.

Run:   python app.py
Open:  http://localhost:8001/metrics   (8000 is used by the project API)

It exposes:
  - The three metrics from the class guide (for the Grafana panels):
      app_requests_total, app_success_total, app_temperature
  - Metrics from the project's src/monitoring/metrics_exporter.py, filled
    with simulated vulnerability prediction traffic.
"""

import os
import random
import sys
import time

from prometheus_client import Counter, Gauge, start_http_server

# Let us import the project's metric definitions from src/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.monitoring.metrics_exporter import (  # noqa: E402
    ACTIVE_CHAMPION_VERSION,
    DATA_DRIFT_SCORE,
    PREDICTION_LATENCY,
    PREDICTION_REQUESTS,
    RISK_SCORE_GAUGE,
    SEVERITY_PREDICTIONS,
)

# ── Guide metrics ────────────────────────────────────────────────────────────
REQUESTS = Counter("app_requests", "Total requests received")        # -> app_requests_total
SUCCESS = Counter("app_success", "Total successful operations")      # -> app_success_total
TEMPERATURE = Gauge("app_temperature", "Current temperature (simulated)")

SEVERITIES = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
MODEL_VERSION = "1"


def simulate_prediction() -> None:
    """Pretend one vulnerability severity prediction happened."""
    REQUESTS.inc()
    start = time.perf_counter()
    time.sleep(random.uniform(0.005, 0.05))  # fake model latency

    ok = random.random() > 0.05  # ~95% success
    status = "success" if ok else "error"
    PREDICTION_REQUESTS.labels(model_version=MODEL_VERSION, status=status).inc()
    PREDICTION_LATENCY.labels(model_version=MODEL_VERSION).observe(time.perf_counter() - start)

    if ok:
        SUCCESS.inc()
        SEVERITY_PREDICTIONS.labels(
            severity_tier=random.choices(SEVERITIES, weights=[30, 40, 20, 10])[0]
        ).inc()
        RISK_SCORE_GAUGE.observe(random.uniform(0, 100))


def main() -> None:
    start_http_server(8001)
    print("Metrics available at http://localhost:8001/metrics  (Ctrl+C to stop)")

    ACTIVE_CHAMPION_VERSION.set(int(MODEL_VERSION))
    while True:
        for _ in range(random.randint(1, 5)):
            simulate_prediction()
        TEMPERATURE.set(round(random.uniform(20, 35), 2))
        DATA_DRIFT_SCORE.set(round(random.uniform(0, 0.4), 3))
        time.sleep(2)


if __name__ == "__main__":
    main()
