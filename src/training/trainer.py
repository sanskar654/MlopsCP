"""
MLflow-Tracked Training Orchestrator (Modern MLOps & Symptom-Based)
===================================================================
Manages the full training lifecycle:
1. Load real vulnerability dataset and split temporally (no leakage)
2. Handle class imbalance using Class Weights (Cost-sensitive learning instead of SMOTE)
3. Train XGBoost classifier with isotonic probability calibration
4. Compute comprehensive evaluation: Macro F1, Recall per class (especially CRITICAL),
   Confusion Matrix, Expected Calibration Error (ECE), and Error Analysis.
5. Log parameters, metrics, SHAP artifacts, model signature, and input examples to MLflow.
6. Register model with modern MLflow aliases (@champion, @challenger).
7. Run quality gate assessment.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import mlflow
    import mlflow.sklearn
    from mlflow.models import infer_signature
    from mlflow.tracking import MlflowClient
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False
    mlflow = None

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight

from src.features.feature_engineer import ALL_FEATURES, TARGET_COLUMN
from src.models.explainer import SHAPExplainer
from src.models.severity_classifier import SeverityClassifier
from src.training.cross_validation import TemporalCrossValidator
from src.training.quality_gate import ModelQualityGate

logger = logging.getLogger(__name__)


class VulnerabilityTrainer:
    """
    End-to-end training orchestrator with MLflow tracking and alias management.
    """

    def __init__(
        self,
        experiment_name: str = "vulnerability-severity-prediction",
        mlflow_tracking_uri: Optional[str] = None,
        tracking_uri: Optional[str] = None,
        model_config: Optional[Dict[str, Any]] = None,
        quality_gate_config: Optional[Dict[str, Any]] = None,
    ):
        uri = tracking_uri or mlflow_tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
        self.mlflow_tracking_uri = uri
        self.experiment_name = experiment_name
        self.model_config = model_config or {}
        self.quality_gate = ModelQualityGate(quality_gate_config)

        if HAS_MLFLOW:
            try:
                mlflow.set_tracking_uri(self.mlflow_tracking_uri)
                mlflow.set_experiment(experiment_name)
                logger.info(f"MLflow tracking initialized at {self.mlflow_tracking_uri}")
            except Exception as e:
                logger.warning(f"MLflow setup warning: {e}")

    def train(
        self,
        df_features: pd.DataFrame,
        model_name: str = "vulnerability-severity-classifier",
        data_version: str = "2.0-real-nvd",
        challenger: bool = False,
    ) -> Tuple[str, Dict[str, Any]]:
        run_name = f"{'challenger' if challenger else 'champion'}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        run_id = str(uuid.uuid4())[:8]

        def _log_param(k, v):
            if HAS_MLFLOW:
                try: mlflow.log_param(k, v)
                except Exception: pass

        def _log_params(d):
            if HAS_MLFLOW:
                try: mlflow.log_params(d)
                except Exception: pass

        def _log_metric(k, v):
            if HAS_MLFLOW:
                try: mlflow.log_metric(k, v)
                except Exception: pass

        if HAS_MLFLOW:
            try:
                active_run = mlflow.start_run(run_name=run_name)
                run_id = active_run.info.run_id
            except Exception as e:
                logger.warning(f"MLflow run start warning: {e}")

        logger.info(f"Training run ID: {run_id}")

        # ── Log configuration ─────────────────────────────────────────
        _log_params({
            "data_version": data_version,
            "n_samples": len(df_features),
            "n_features": len(ALL_FEATURES),
            "model_type": "XGBoost+IsotonicCalibration",
            "imbalance_strategy": "Class_Weights_Cost_Sensitive",
            "cv_strategy": "temporal_split",
            "feature_version": "2.0-leakage-free",
        })

        # ── Data preparation ──────────────────────────────────────────
        logger.info("Preparing training data...")
        df_labeled = df_features[
            df_features[TARGET_COLUMN].notna()
            & (df_features[TARGET_COLUMN] >= 0)
        ].copy()

        logger.info(f"Labeled samples: {len(df_labeled)} / {len(df_features)} total")
        _log_metric("n_labeled_samples", len(df_labeled))

        dist = df_labeled[TARGET_COLUMN].value_counts().to_dict()
        for cls, count in dist.items():
            _log_metric(f"class_{cls}_count", count)

        # ── Temporal Split ────────────────────────────────────────────
        logger.info("Performing temporal train/val/test split...")
        cv = TemporalCrossValidator()
        train_df, val_df, test_df = cv.temporal_split(df_labeled)

        cols_to_use = [c for c in ALL_FEATURES if c in train_df.columns]
        X_train = train_df[cols_to_use].fillna(0)
        y_train = train_df[TARGET_COLUMN].astype(int)
        X_val = val_df[cols_to_use].fillna(0)
        y_val = val_df[TARGET_COLUMN].astype(int)
        X_test = test_df[cols_to_use].fillna(0)
        y_test = test_df[TARGET_COLUMN].astype(int)

        _log_params({
            "train_size": len(X_train),
            "val_size": len(X_val),
            "test_size": len(X_test),
        })

        # ── Class Balancing via Cost-Sensitive Weights (No SMOTE on text) ────
        logger.info("Applying balanced class sample weights to loss function...")
        sample_weights = compute_sample_weight("balanced", y_train)
        _log_param("imbalance_method", "balanced_sample_weights")

        # ── Train Model ───────────────────────────────────────────────
        logger.info("Training calibrated severity classifier...")
        t0 = time.time()
        classifier = SeverityClassifier(config=self.model_config or None)
        classifier.build_pipeline()
        classifier.fit(
            X_train,
            y_train,
            X_val=X_val,
            y_val=y_val,
            sample_weights=sample_weights,
        )
        train_time = time.time() - t0
        _log_metric("training_time_seconds", train_time)
        logger.info(f"Training completed in {train_time:.1f}s")

        # ── Comprehensive Evaluation ──────────────────────────────────
        logger.info("Evaluating model on test set...")
        metrics = classifier.evaluate(X_test, y_test)

        # Compute error analysis: examine confusion pairs and reasons
        y_test_pred = classifier.predict(X_test)
        error_mask = (y_test.values != y_test_pred)
        n_errors = int(np.sum(error_mask))
        error_rate = float(n_errors / max(len(y_test), 1))
        _log_metric("error_rate", error_rate)
        _log_metric("n_misclassifications", n_errors)

        # Log metrics to MLflow
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                _log_metric(metric_name, value)

        # Inference Latency
        latency_p99 = self._measure_latency(classifier, X_test)
        _log_metric("latency_p99_ms", latency_p99)
        metrics["latency_p99_ms"] = latency_p99

        # SHAP Importance
        logger.info("Computing SHAP feature importance...")
        try:
            explainer = SHAPExplainer(classifier)
            bg_sample = X_train.sample(min(150, len(X_train)), random_state=42)
            explainer.initialize(bg_sample)
            global_importance = explainer.global_feature_importance(X_test, n_samples=80)

            importance_df = pd.DataFrame(global_importance)
            importance_path = "data/monitoring/feature_importance.csv"
            Path(importance_path).parent.mkdir(parents=True, exist_ok=True)
            importance_df.to_csv(importance_path, index=False)
            if HAS_MLFLOW:
                try: mlflow.log_artifact(importance_path, "explainability")
                except Exception: pass
        except Exception as e:
            logger.warning(f"SHAP importance computation skipped/failed: {e}")

        # ── MLflow Model Logging with Signature, Input Example & Aliases ────
        logger.info("Logging model to MLflow with signature & input example...")
        if HAS_MLFLOW:
            try:
                input_example = X_test.head(2)
                pred_example = classifier.predict_proba(input_example)
                signature = infer_signature(input_example, pred_example)

                model_info = mlflow.sklearn.log_model(
                    sk_model=classifier.calibrated_model,
                    artifact_path="model",
                    registered_model_name=model_name,
                    signature=signature,
                    input_example=input_example,
                )

                # Set MLflow Model Alias (@champion or @challenger)
                try:
                    client = MlflowClient()
                    assigned_alias = "challenger" if challenger else "champion"
                    latest_versions = client.get_latest_versions(model_name)
                    if latest_versions:
                        target_v = latest_versions[0].version
                        client.set_registered_model_alias(model_name, assigned_alias, target_v)
                        logger.info(f"Registered model alias @{assigned_alias} assigned to version {target_v}")
                except Exception as e_alias:
                    logger.warning(f"Could not assign model alias: {e_alias}")

            except Exception as e:
                logger.warning(f"MLflow log_model warning: {e}")

        # ── Quality Gate ──────────────────────────────────────────────
        logger.info("Running model quality gate...")
        gate_result = self.quality_gate.evaluate(metrics)
        _log_param("quality_gate_passed", gate_result["passed"])

        if HAS_MLFLOW:
            try: mlflow.end_run()
            except Exception: pass

        if gate_result["passed"]:
            logger.info("Quality gate PASSED - Model eligible for production champion alias")
        else:
            logger.warning(f"Quality gate warnings/failures: {gate_result.get('failures', [])}")

        metrics["quality_gate"] = gate_result
        metrics["run_id"] = run_id
        metrics["model_name"] = model_name

        return run_id, metrics

    def train_and_evaluate(
        self,
        data_path: str = "data/processed/features.parquet",
        model_name: str = "vulnerability-severity-classifier",
    ) -> Tuple[str, Dict[str, Any], Any]:
        """Convenience runner loading features and triggering training."""
        if not os.path.exists(data_path):
            csv_path = data_path.replace(".parquet", ".csv")
            if os.path.exists(csv_path):
                data_path = csv_path
            else:
                raise FileNotFoundError(f"Feature file not found at {data_path}")

        if data_path.endswith(".parquet"):
            df = pd.read_parquet(data_path)
        else:
            df = pd.read_csv(data_path)

        run_id, metrics = self.train(df_features=df, model_name=model_name, challenger=False)
        return run_id, metrics, None

    def _measure_latency(
        self,
        classifier: SeverityClassifier,
        X_test: pd.DataFrame,
        n_warmup: int = 5,
        n_trials: int = 50,
    ) -> float:
        """Measure p99 inference latency in milliseconds."""
        sample = X_test.head(1)
        times = []
        for _ in range(n_warmup):
            classifier.predict_proba(sample)
        for _ in range(n_trials):
            t0 = time.perf_counter()
            classifier.predict_proba(sample)
            times.append((time.perf_counter() - t0) * 1000)
        p99 = float(np.percentile(times, 99))
        return round(p99, 2)
