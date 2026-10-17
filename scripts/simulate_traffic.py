"""
Traffic and Load Testing Simulation Script
==========================================
Simulates live inference traffic to the FastAPI serving layer,
records latency percentiles (p50, p90, p95, p99), requests per second (RPS),
and error rates.
Supports both live HTTP endpoints and in-process TestClient execution.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("load_test")

SAMPLE_CVE_PAYLOADS = [
    {
        "cve_id": "CVE-2024-38063",
        "description": "Windows TCP/IP Remote Code Execution Vulnerability allowing unauthenticated attacker to send crafted packets.",
        "cwe_ids": ["CWE-120", "CWE-121"],
        "epss_score": 0.942,
        "epss_percentile": 0.991,
        "in_cisa_kev": True,
        "year_published": 2024,
        "asset_criticality": 0.95,
        "exposure_score": 0.90,
    },
    {
        "cve_id": "CVE-2023-38606",
        "description": "Apple WebKit state management vulnerability allowing arbitrary code execution when processing web content.",
        "cwe_ids": ["CWE-20"],
        "epss_score": 0.812,
        "epss_percentile": 0.975,
        "in_cisa_kev": True,
        "year_published": 2023,
        "asset_criticality": 0.85,
        "exposure_score": 0.80,
    },
    {
        "cve_id": "CVE-2024-21626",
        "description": "runc container breakout via file descriptor leak in workdir allowing internal guest to escape to host.",
        "cwe_ids": ["CWE-403"],
        "epss_score": 0.450,
        "epss_percentile": 0.910,
        "in_cisa_kev": False,
        "year_published": 2024,
        "asset_criticality": 0.70,
        "exposure_score": 0.50,
    },
    {
        "cve_id": "CVE-2024-0015",
        "description": "Improper input validation in search filter leading to reflected cross-site scripting (XSS).",
        "cwe_ids": ["CWE-79"],
        "epss_score": 0.015,
        "epss_percentile": 0.220,
        "in_cisa_kev": False,
        "year_published": 2024,
        "asset_criticality": 0.40,
        "exposure_score": 0.30,
    },
    {
        "cve_id": "CVE-2024-0099",
        "description": "Null pointer dereference in diagnostic parser causes denial of service crash when processing malformed packets.",
        "cwe_ids": ["CWE-476"],
        "epss_score": 0.040,
        "epss_percentile": 0.350,
        "in_cisa_kev": False,
        "year_published": 2024,
        "asset_criticality": 0.50,
        "exposure_score": 0.40,
    },
]


def run_benchmark(
    base_url: str = "http://localhost:8000",
    n_requests: int = 150,
    endpoint: str = "/api/v1/risk-score",
    use_client_fallback: bool = True,
) -> Dict[str, Any]:
    """
    Executes benchmark requests and computes latency percentiles.
    """
    logger.info(
        "Starting load benchmark: %d requests to %s%s...",
        n_requests,
        base_url,
        endpoint,
    )

    latencies_ms: List[float] = []
    success_count = 0
    error_count = 0

    client = None
    # Check if server is running on HTTP
    server_online = False
    try:
        req = urllib.request.Request(f"{base_url}/health")
        with urllib.request.urlopen(req, timeout=1) as resp:
            if resp.status == 200:
                server_online = True
                logger.info("Connected to live server at %s", base_url)
    except Exception:
        server_online = False

    if not server_online and use_client_fallback:
        logger.info(
            "Live server not detected. Running load test in-process via FastAPI TestClient..."
        )
        from fastapi.testclient import TestClient
        from src.serving.api.main import app

        client = TestClient(app)

    total_start = time.perf_counter()

    for i in range(n_requests):
        payload = random.choice(SAMPLE_CVE_PAYLOADS).copy()
        # vary cve_id slightly
        payload["cve_id"] = f"{payload['cve_id']}-SIM-{i:03d}"

        t0 = time.perf_counter()
        if server_online:
            try:
                data = json.dumps(payload).encode("utf-8")
                http_req = urllib.request.Request(
                    f"{base_url}{endpoint}",
                    data=data,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(http_req, timeout=5) as resp:
                    elapsed = (time.perf_counter() - t0) * 1000
                    latencies_ms.append(elapsed)
                    if resp.status == 200:
                        success_count += 1
                    else:
                        error_count += 1
            except Exception:
                error_count += 1
        elif client is not None:
            try:
                resp = client.post(endpoint, json=payload)
                elapsed = (time.perf_counter() - t0) * 1000
                latencies_ms.append(elapsed)
                if resp.status_code == 200:
                    success_count += 1
                else:
                    error_count += 1
            except Exception:
                error_count += 1

    total_duration = time.perf_counter() - total_start
    rps = float(n_requests / max(total_duration, 0.001))

    latencies = np.array(latencies_ms) if latencies_ms else np.array([0.0])
    p50 = float(np.percentile(latencies, 50))
    p90 = float(np.percentile(latencies, 90))
    p95 = float(np.percentile(latencies, 95))
    p99 = float(np.percentile(latencies, 99))
    mean_lat = float(np.mean(latencies))
    min_lat = float(np.min(latencies))
    max_lat = float(np.max(latencies))

    results = {
        "total_requests": n_requests,
        "successful_requests": success_count,
        "failed_requests": error_count,
        "success_rate_percent": round(
            float(success_count / max(n_requests, 1)) * 100, 2
        ),
        "total_duration_seconds": round(total_duration, 2),
        "throughput_rps": round(rps, 2),
        "latency_min_ms": round(min_lat, 2),
        "latency_mean_ms": round(mean_lat, 2),
        "latency_p50_ms": round(p50, 2),
        "latency_p90_ms": round(p90, 2),
        "latency_p95_ms": round(p95, 2),
        "latency_p99_ms": round(p99, 2),
        "latency_max_ms": round(max_lat, 2),
    }

    print("\n" + "=" * 65)
    print("             API LOAD TEST & BENCHMARK RESULTS")
    print("=" * 65)
    for k, v in results.items():
        print(f"  {k:30s}: {v}")
    print("=" * 65 + "\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default="http://localhost:8000")
    parser.add_argument("--count", type=int, default=150)
    parser.add_argument("--endpoint", type=str, default="/api/v1/risk-score")
    args = parser.parse_args()

    run_benchmark(base_url=args.url, n_requests=args.count, endpoint=args.endpoint)
