"""
SHAP Explainer
==============
Generates SHAP-based explanations for individual predictions and global
feature importance using TreeExplainer for XGBoost models.

Provides:
- Per-prediction SHAP waterfall data (top N features with contributions)
- Global feature importance (mean |SHAP|)
- Human-readable explanation text
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SEVERITY_LABELS = ["NONE", "LOW", "MEDIUM", "HIGH", "CRITICAL"]


class SHAPExplainer:
    """
    SHAP TreeExplainer wrapper for vulnerability severity predictions.

    Usage:
        explainer = SHAPExplainer(classifier)
        explanation = explainer.explain_single(features_row, predicted_class=4)
    """

    def __init__(self, classifier: Any):
        """
        Args:
            classifier: Fitted SeverityClassifier instance
        """
        self.classifier = classifier
        self._shap_explainer = None
        self._background_data: pd.DataFrame | None = None

    def initialize(self, background_data: pd.DataFrame) -> None:
        """
        Initialize SHAP explainer with background dataset.

        Args:
            background_data: Representative sample of training data (100–500 rows)
        """
        try:
            import shap
            base_model = self.classifier.get_base_model()
            if base_model is None:
                raise ValueError("Classifier must be fitted before initializing explainer")

            bg = background_data.fillna(0)
            self._shap_explainer = shap.TreeExplainer(
                base_model,
                data=bg,
                feature_perturbation="interventional",
            )
            self._background_data = background_data
            logger.info(f"SHAP explainer initialized with {len(bg)} background samples")
        except ImportError:
            logger.warning("SHAP not installed. Explanations will be unavailable.")

    def explain_single(
        self,
        X: pd.DataFrame,
        predicted_class: int,
        top_n: int = 10,
    ) -> dict[str, Any]:
        """
        Generate SHAP explanation for a single prediction.

        Args:
            X: Single-row feature DataFrame
            predicted_class: The predicted severity class (0–4)
            top_n: Number of top features to return

        Returns:
            Dict with feature contributions, base value, and explanation text
        """
        if self._shap_explainer is None:
            return self._fallback_explanation(X, predicted_class)

        try:
            import shap
            X_filled = X.fillna(0)
            shap_values = self._shap_explainer.shap_values(X_filled)

            # shap_values shape: [n_samples, n_features, n_classes]
            if isinstance(shap_values, list):
                # Older SHAP: list of arrays per class
                class_shap = shap_values[predicted_class][0]
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                class_shap = shap_values[0, :, predicted_class]
            else:
                class_shap = shap_values[0] if shap_values.ndim == 2 else shap_values

            feature_names = list(X.columns)
            contributions = [
                {
                    "feature": name,
                    "shap_value": float(val),
                    "feature_value": float(X_filled.iloc[0][name]),
                    "direction": "positive" if val > 0 else "negative",
                }
                for name, val in zip(feature_names, class_shap)
            ]

            # Sort by absolute SHAP value
            contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
            top_contributions = contributions[:top_n]

            # Base value
            if hasattr(self._shap_explainer, "expected_value"):
                ev = self._shap_explainer.expected_value
                if isinstance(ev, (list, np.ndarray)):
                    base_value = float(ev[predicted_class])
                else:
                    base_value = float(ev)
            else:
                base_value = 0.0

            return {
                "predicted_class": predicted_class,
                "predicted_label": SEVERITY_LABELS[predicted_class],
                "base_value": base_value,
                "top_features": top_contributions,
                "explanation_text": self._generate_text(
                    top_contributions, predicted_class
                ),
                "total_shap_sum": float(sum(c["shap_value"] for c in contributions)),
            }

        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}")
            return self._fallback_explanation(X, predicted_class)

    def global_feature_importance(
        self,
        X: pd.DataFrame,
        n_samples: int = 200,
        top_n: int = 20,
    ) -> list[dict[str, float]]:
        """
        Compute global feature importance as mean |SHAP| across samples.

        Args:
            X: Representative dataset
            n_samples: Max samples to use (for speed)
            top_n: Number of top features to return
        """
        if self._shap_explainer is None:
            return []

        try:
            sample = X.fillna(0).sample(min(n_samples, len(X)), random_state=42)
            shap_values = self._shap_explainer.shap_values(sample)

            if isinstance(shap_values, list):
                # Average across all classes
                mean_abs = np.mean(
                    [np.abs(sv).mean(axis=0) for sv in shap_values], axis=0
                )
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                mean_abs = np.abs(shap_values).mean(axis=(0, 2))
            else:
                mean_abs = np.abs(shap_values).mean(axis=0)

            importance = sorted(
                [
                    {"feature": name, "importance": float(val)}
                    for name, val in zip(X.columns, mean_abs)
                ],
                key=lambda x: x["importance"],
                reverse=True,
            )
            return importance[:top_n]

        except Exception as e:
            logger.error(f"Global SHAP importance failed: {e}")
            return []

    def _generate_text(
        self,
        contributions: list[dict],
        predicted_class: int,
    ) -> str:
        """Generate human-readable explanation text."""
        severity = SEVERITY_LABELS[predicted_class]
        positive = [c for c in contributions[:5] if c["shap_value"] > 0]
        negative = [c for c in contributions[:5] if c["shap_value"] < 0]

        lines = [f"Predicted as **{severity}** severity."]

        if positive:
            factors = ", ".join(
                f"`{c['feature']}` (+{c['shap_value']:.3f})"
                for c in positive[:3]
            )
            lines.append(f"Factors **increasing** severity: {factors}.")

        if negative:
            factors = ", ".join(
                f"`{c['feature']}` ({c['shap_value']:.3f})"
                for c in negative[:3]
            )
            lines.append(f"Factors **decreasing** severity: {factors}.")

        return " ".join(lines)

    def _fallback_explanation(
        self, X: pd.DataFrame, predicted_class: int
    ) -> dict[str, Any]:
        """Return feature importance-based explanation when SHAP unavailable."""
        fi = self.classifier.get_feature_importance()
        top = list(fi.items())[:10]
        return {
            "predicted_class": predicted_class,
            "predicted_label": SEVERITY_LABELS[predicted_class],
            "base_value": None,
            "top_features": [
                {"feature": k, "shap_value": None, "importance": v}
                for k, v in top
            ],
            "explanation_text": f"Predicted {SEVERITY_LABELS[predicted_class]}. "
                               f"SHAP unavailable; showing feature importance.",
            "total_shap_sum": None,
        }
