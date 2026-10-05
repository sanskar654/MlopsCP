"""
Performance Monitor Module
==========================
Tracks model operational and statistical performance in production:
- Latency (p50, p95, p99)
- Throughput (QPS)
- Prediction distribution vs Ground Truth (when labels arrive)
- Realized accuracy, F1, and critical-class recall
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import classification_report, f1_score, recall_score

logger = logging.getLogger("mlops.monitoring.performance")


class PerformanceTracker:
    """Tracks latency, throughput, prediction distribution, and post-hoc accuracy."""

    def __init__(self, window_size: int = 5000):
        self.window_size = window_size
        self.latencies_ms: List[float] = []
        self.predictions: List[str] = []
        self.true_labels: List[str] = []
        self.timestamps: List[datetime] = []

    def record_inference(
        self, latency_ms: float, prediction: str, true_label: Optional[str] = None
    ) -> None:
        """Record a single inference event."""
        self.latencies_ms.append(latency_ms)
        self.predictions.append(prediction)
        self.timestamps.append(datetime.now(timezone.utc))

        if true_label is not None:
            self.true_labels.append(true_label)

        # Evict oldest if exceeding window size
        if len(self.latencies_ms) > self.window_size:
            self.latencies_ms.pop(0)
            self.predictions.pop(0)
            self.timestamps.pop(0)
            if len(self.true_labels) > self.window_size:
                self.true_labels.pop(0)

    def get_latency_metrics(self) -> Dict[str, float]:
        """Compute latency percentiles."""
        if not self.latencies_ms:
            return {"p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "mean_ms": 0.0}

        arr = np.array(self.latencies_ms)
        return {
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)),
            "mean_ms": float(np.mean(arr)),
            "sample_count": len(arr),
        }

    def get_prediction_distribution(self) -> Dict[str, float]:
        """Compute class distribution of recent predictions."""
        if not self.predictions:
            return {}
        total = len(self.predictions)
        counts: Dict[str, int] = {}
        for p in self.predictions:
            counts[p] = counts.get(p, 0) + 1
        return {k: round(v / total, 4) for k, v in counts.items()}

    def evaluate_ground_truth(self) -> Optional[Dict[str, Any]]:
        """Evaluate accuracy and recall if ground-truth labels are available."""
        if len(self.true_labels) < 20 or len(self.true_labels) != len(
            self.predictions[: len(self.true_labels)]
        ):
            return None

        y_true = self.true_labels
        y_pred = self.predictions[: len(self.true_labels)]

        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        critical_recall = recall_score(
            [1 if y == "CRITICAL" else 0 for y in y_true],
            [1 if y == "CRITICAL" else 0 for y in y_pred],
            zero_division=0,
        )

        return {
            "macro_f1": float(macro_f1),
            "critical_recall": float(critical_recall),
            "evaluated_samples": len(y_true),
            "report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
        }
