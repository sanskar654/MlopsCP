"""
Alerting and Notification Engine
================================
Dispatches automated alerts on drift detection, model performance degradation,
or high-criticality zero-day vulnerability spikes.
Supports Slack webhooks, PagerDuty, and structured logs.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import urllib.request
import urllib.error

logger = logging.getLogger("mlops.monitoring.alerting")


class AlertManager:
    """Manages alert routing, deduplication, and notification dispatching."""

    def __init__(self, webhook_url: Optional[str] = None):
        self.webhook_url = webhook_url or os.getenv("SLACK_WEBHOOK_URL")
        self.alert_history: List[Dict[str, Any]] = []

    def dispatch_alert(
        self,
        severity: str,
        title: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Dispatches an alert to configured webhook and audit log."""
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "severity": severity.upper(),
            "title": title,
            "message": message,
            "metadata": metadata or {},
        }
        self.alert_history.append(payload)
        logger.warning(
            "ALERT [%s] %s: %s (Metadata: %s)",
            severity.upper(),
            title,
            message,
            metadata,
        )

        if not self.webhook_url:
            return True

        try:
            req_data = json.dumps(
                {
                    "text": f"*{severity.upper()}*: {title}\n{message}\n```\n{json.dumps(metadata or {}, indent=2)}\n```"
                }
            ).encode("utf-8")

            req = urllib.request.Request(
                self.webhook_url,
                data=req_data,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=5) as response:
                return response.status == 200
        except Exception as e:
            logger.error("Failed to post alert to webhook: %s", str(e))
            return False

    def check_drift_alert(self, drift_report: Dict[str, Any]) -> Optional[bool]:
        """Trigger alert if drift is detected."""
        if drift_report.get("dataset_drift", False):
            drift_share = drift_report.get("drift_share", 0.0) * 100
            drifted_count = drift_report.get("drifted_features_count", 0)
            return self.dispatch_alert(
                severity="HIGH",
                title="Data Drift Detected - Retraining Recommended",
                message=f"Dataset drift triggered: {drifted_count} features drifted ({drift_share:.1f}% share).",
                metadata={"drift_share": drift_share, "drifted_count": drifted_count},
            )
        return None

    def check_latency_alert(
        self, latency_metrics: Dict[str, float], threshold_p95_ms: float = 200.0
    ) -> Optional[bool]:
        """Trigger alert if p95 inference latency exceeds SLA threshold."""
        p95 = latency_metrics.get("p95_ms", 0.0)
        if p95 > threshold_p95_ms:
            return self.dispatch_alert(
                severity="MEDIUM",
                title="Inference Latency SLA Breach",
                message=f"P95 latency is {p95:.1f}ms (threshold: {threshold_p95_ms}ms).",
                metadata=latency_metrics,
            )
        return None
