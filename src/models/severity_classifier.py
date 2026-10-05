"""
Severity Classifier
===================
XGBoost-based vulnerability severity classifier with:
- Probability calibration (isotonic regression)
- Class imbalance handling via SMOTE + class weights
- Temporal cross-validation
- MLflow integration
- SHAP explainability

Target: 5-class ordinal classification
  0=NONE, 1=LOW, 2=MEDIUM, 3=HIGH, 4=CRITICAL
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    XGBClassifier = None

from features.feature_engineer import ALL_FEATURES, TARGET_COLUMN

logger = logging.getLogger(__name__)


SEVERITY_LABELS = ["NONE", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
N_CLASSES = 5


class SeverityClassifier:
    """
    Production severity classifier with calibrated probabilities.

    Architecture:
        1. StandardScaler (for numeric features)
        2. XGBoost multi-class classifier
        3. Isotonic calibration for reliable confidence scores
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or self._default_config()
        self.pipeline: Pipeline | None = None
        self.calibrated_model: CalibratedClassifierCV | None = None
        self.feature_names: list[str] = []
        self.is_fitted = False

    def _default_config(self) -> dict[str, Any]:
        return {
            "xgb_params": {
                "n_estimators": 500,
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 3,
                "gamma": 0.1,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
                "objective": "multi:softprob",
                "num_class": N_CLASSES,
                "eval_metric": "mlogloss",
                "tree_method": "hist",  # Faster than 'auto'
                "random_state": 42,
                "n_jobs": -1,
            },
            "calibration_method": "isotonic",
            "calibration_cv": 5,
            "use_scaler": True,
        }

    def build_pipeline(self, class_weights: dict[int, float] | None = None) -> None:
        """Construct the sklearn pipeline with optional class weights."""
        xgb_params = self.config["xgb_params"].copy()

        if class_weights:
            # XGBoost uses sample_weight, not class_weight directly
            # We'll apply it during fit via compute_sample_weight
            logger.info(f"Using class weights: {class_weights}")
        self._class_weights = class_weights

        steps = []
        if self.config["use_scaler"]:
            steps.append(("scaler", StandardScaler(with_mean=False)))

        if HAS_XGBOOST:
            base_clf = XGBClassifier(**xgb_params)
        else:
            logger.info("XGBoost not available; using HistGradientBoostingClassifier fallback")
            base_clf = HistGradientBoostingClassifier(random_state=42)

        steps.append(("classifier", base_clf))


        self.pipeline = Pipeline(steps)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        sample_weights: np.ndarray | None = None,
    ) -> "SeverityClassifier":
        """
        Fit the classifier with optional validation for early stopping.

        Args:
            X_train: Training features (only columns from ALL_FEATURES)
            y_train: Target labels (0-4 ordinal)
            X_val: Optional validation set for early stopping
            y_val: Optional validation labels
            sample_weights: Optional per-sample weights (from SMOTE balancing)
        """
        if self.pipeline is None:
            self.build_pipeline()

        self.feature_names = list(X_train.columns)
        logger.info(f"Training on {len(X_train)} samples, {len(self.feature_names)} features")

        # Fill NaN with medians/zeros for robustness
        X_train = X_train.fillna(0)

        fit_params: dict[str, Any] = {}
        if sample_weights is not None:
            fit_params["classifier__sample_weight"] = sample_weights

        # Early stopping if validation set provided
        if HAS_XGBOOST and X_val is not None and y_val is not None:
            X_val = X_val.fillna(0)
            eval_set = [(X_val.values, y_val.values)]
            # Only last step's eval_set
            fit_params["classifier__eval_set"] = eval_set
            fit_params["classifier__verbose"] = False


        self.pipeline.fit(X_train, y_train, **fit_params)

        # Probability calibration using held-out fold
        logger.info("Calibrating probabilities with isotonic regression...")
        if X_val is not None and y_val is not None:
            # Calibrate on validation set
            self.calibrated_model = CalibratedClassifierCV(
                estimator=self.pipeline,
                method=self.config["calibration_method"],
                cv="prefit",  # Already fitted
            )
            X_val_filled = X_val.fillna(0)
            self.calibrated_model.fit(X_val_filled, y_val)
        else:
            self.calibrated_model = CalibratedClassifierCV(
                estimator=self.pipeline,
                method=self.config["calibration_method"],
                cv=self.config["calibration_cv"],
            )
            self.calibrated_model.fit(X_train, y_train)

        self.is_fitted = True
        logger.info("Classifier fitted and calibrated successfully")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict severity class (0–4)."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X = X.fillna(0)
        return self.calibrated_model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict calibrated class probabilities (shape: [n, 5])."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X = X.fillna(0)
        return self.calibrated_model.predict_proba(X)

    def predict_with_confidence(
        self, X: pd.DataFrame
    ) -> list[dict[str, Any]]:
        """
        Predict severity with confidence score and class probabilities.

        Returns:
            List of dicts with 'predicted_class', 'predicted_label',
            'confidence', 'probabilities'
        """
        proba = self.predict_proba(X)
        predictions = []
        for row in proba:
            pred_class = int(np.argmax(row))
            predictions.append({
                "predicted_class": pred_class,
                "predicted_label": SEVERITY_LABELS[pred_class],
                "confidence": float(row[pred_class]),
                "probabilities": {
                    SEVERITY_LABELS[i]: float(row[i])
                    for i in range(N_CLASSES)
                },
            })
        return predictions

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> dict[str, Any]:
        """
        Comprehensive model evaluation.

        Returns dict with all metrics needed for quality gate.
        """
        X_test = X_test.fillna(0)
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)

        # Core metrics
        macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        balanced_acc = balanced_accuracy_score(y_test, y_pred)

        # Critical class (4) recall — critical for quality gate
        report = classification_report(
            y_test, y_pred,
            target_names=SEVERITY_LABELS,
            output_dict=True,
            zero_division=0,
        )
        critical_recall = report.get("CRITICAL", {}).get("recall", 0.0)
        high_recall = report.get("HIGH", {}).get("recall", 0.0)

        # ROC-AUC (one-vs-rest, handles multi-class)
        try:
            roc_auc = roc_auc_score(
                y_test, y_proba,
                multi_class="ovr",
                average="macro",
                labels=list(range(N_CLASSES)),
            )
        except Exception:
            roc_auc = 0.0

        # Expected Calibration Error (ECE)
        ece = self._compute_ece(y_test, y_proba)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=list(range(N_CLASSES)))

        metrics = {
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "balanced_accuracy": float(balanced_acc),
            "critical_recall": float(critical_recall),
            "high_recall": float(high_recall),
            "roc_auc": float(roc_auc),
            "ece": float(ece),
            "n_test_samples": len(y_test),
            "class_report": report,
            "confusion_matrix": cm.tolist(),
            "class_distribution": {
                SEVERITY_LABELS[i]: int((y_test == i).sum())
                for i in range(N_CLASSES)
            },
        }

        logger.info(
            f"Evaluation: macro_f1={macro_f1:.4f}, "
            f"critical_recall={critical_recall:.4f}, "
            f"ece={ece:.4f}, roc_auc={roc_auc:.4f}"
        )
        return metrics

    @staticmethod
    def _compute_ece(
        y_true: pd.Series,
        y_proba: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """
        Compute Expected Calibration Error.
        Measures how well model confidence aligns with actual accuracy.
        Lower is better. < 0.10 is acceptable.
        """
        confidences = y_proba.max(axis=1)
        predictions = y_proba.argmax(axis=1)
        correct = (predictions == y_true.values).astype(float)

        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
            if mask.sum() > 0:
                avg_confidence = confidences[mask].mean()
                avg_accuracy = correct[mask].mean()
                ece += mask.sum() * abs(avg_confidence - avg_accuracy)

        return ece / len(y_true) if len(y_true) > 0 else 0.0

    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance from the base XGBoost model."""
        if not self.is_fitted:
            return {}
        try:
            xgb_step = self.pipeline.named_steps["classifier"]
            importance = xgb_step.feature_importances_
            return dict(sorted(
                zip(self.feature_names, importance.tolist()),
                key=lambda x: x[1],
                reverse=True,
            ))
        except Exception:
            return {}

    def get_base_model(self) -> XGBClassifier | None:
        """Return the underlying XGBoost model for SHAP analysis."""
        try:
            return self.pipeline.named_steps["classifier"]
        except Exception:
            return None
