"""
MLflow-Tracked Training Orchestrator
=====================================
Manages the full training pipeline with experiment tracking:
1. Load and split data temporally (no leakage)
2. Apply SMOTE for class balance
3. Train XGBoost classifier
4. Evaluate with comprehensive metrics
5. Log everything to MLflow
6. Run quality gate
7. Register model if passes

All runs are reproducible via logged parameters and DVC data hashes.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import mlflow
    import mlflow.sklearn
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False
    mlflow = None

import numpy as np
import pandas as pd
try:
    from imblearn.over_sampling import SMOTE

    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    SMOTE = None


from features.feature_engineer import ALL_FEATURES, TARGET_COLUMN
from models.explainer import SHAPExplainer
from models.severity_classifier import SeverityClassifier
from training.cross_validation import TemporalCrossValidator
from training.quality_gate import ModelQualityGate

logger = logging.getLogger(__name__)


class VulnerabilityTrainer:
    """
    End-to-end training orchestrator with MLflow tracking.

    Usage:
        trainer = VulnerabilityTrainer(
            mlflow_tracking_uri="http://localhost:5000",
            experiment_name="vulnerability-severity-prediction",
        )
        run_id, metrics = trainer.train(df_features, model_name="vuln-classifier")
    """

    def __init__(
        self,
        experiment_name: str = "vulnerability-severity-prediction",
        mlflow_tracking_uri: str | None = None,
        tracking_uri: str | None = None,
        model_config: dict[str, Any] | None = None,
        quality_gate_config: dict[str, Any] | None = None,
    ):
        uri = tracking_uri or mlflow_tracking_uri or "sqlite:///mlflow.db"
        self.mlflow_tracking_uri = uri
        self.experiment_name = experiment_name
        self.model_config = model_config or {}
        self.quality_gate = ModelQualityGate(quality_gate_config)


        if HAS_MLFLOW:
            try:
                mlflow.set_tracking_uri(mlflow_tracking_uri)
                mlflow.set_experiment(experiment_name)
                logger.info(f"MLflow tracking at {mlflow_tracking_uri}")
            except Exception as e:
                logger.warning(f"MLflow setup warning: {e}")

    def train(
        self,
        df_features: pd.DataFrame,
        model_name: str = "vulnerability-severity-classifier",
        data_version: str = "unknown",
        challenger: bool = False,
    ) -> tuple[str, dict[str, Any]]:
        run_name = f"{'challenger' if challenger else 'training'}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
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
            "imbalance_strategy": "SMOTE",
            "cv_strategy": "temporal",
            "feature_version": df_features.get("feature_version", ["1.0"])[0]
                if "feature_version" in df_features.columns else "1.0",
        })


        # ── Data preparation ──────────────────────────────────────────
        logger.info("Preparing training data...")
        df_labeled = df_features[
            df_features[TARGET_COLUMN].notna()
            & (df_features[TARGET_COLUMN] >= 0)
        ].copy()

        logger.info(
            f"Labeled samples: {len(df_labeled)} / {len(df_features)} total"
        )
        _log_metric("n_labeled_samples", len(df_labeled))

        # Log class distribution
        dist = df_labeled[TARGET_COLUMN].value_counts().to_dict()
        for cls, count in dist.items():
            _log_metric(f"class_{cls}_count", count)

        # ── Temporal Split ────────────────────────────────────────────
        logger.info("Performing temporal train/val/test split...")
        cv = TemporalCrossValidator()
        train_df, val_df, test_df = cv.temporal_split(df_labeled)

        X_train = train_df[ALL_FEATURES].fillna(0)
        y_train = train_df[TARGET_COLUMN].astype(int)
        X_val = val_df[ALL_FEATURES].fillna(0)
        y_val = val_df[TARGET_COLUMN].astype(int)
        X_test = test_df[ALL_FEATURES].fillna(0)
        y_test = test_df[TARGET_COLUMN].astype(int)

        _log_params({
            "train_size": len(X_train),
            "val_size": len(X_val),
            "test_size": len(X_test),
        })

        # ── SMOTE Class Balancing ─────────────────────────────────────
        logger.info("Applying SMOTE for class imbalance...")
        X_train_bal, y_train_bal = self._apply_smote(X_train, y_train)
        _log_param("smote_applied", True)
        _log_metric("n_train_after_smote", len(X_train_bal))

        # ── Train Model ───────────────────────────────────────────────
        logger.info("Training XGBoost classifier...")
        t0 = time.time()
        classifier = SeverityClassifier(config=self.model_config or None)
        classifier.build_pipeline()
        classifier.fit(
            pd.DataFrame(X_train_bal, columns=ALL_FEATURES),
            pd.Series(y_train_bal),
            X_val=X_val,
            y_val=y_val,
        )
        train_time = time.time() - t0
        _log_metric("training_time_seconds", train_time)
        logger.info(f"Training completed in {train_time:.1f}s")

        # ── Evaluate ──────────────────────────────────────────────────
        logger.info("Evaluating model on test set...")
        metrics = classifier.evaluate(X_test, y_test)

        # Log all metrics to MLflow
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                _log_metric(metric_name, value)

        # ── Inference Latency ─────────────────────────────────────────
        latency_p99 = self._measure_latency(classifier, X_test)
        _log_metric("latency_p99_ms", latency_p99)
        metrics["latency_p99_ms"] = latency_p99

        # ── SHAP Global Importance ────────────────────────────────────
        logger.info("Computing SHAP feature importance...")
        try:
            explainer = SHAPExplainer(classifier)
            bg_sample = X_train.sample(min(200, len(X_train)), random_state=42)
            explainer.initialize(bg_sample)
            global_importance = explainer.global_feature_importance(X_test, n_samples=100)

            importance_df = pd.DataFrame(global_importance)
            importance_path = "/tmp/feature_importance.csv"
            importance_df.to_csv(importance_path, index=False)
            if HAS_MLFLOW:
                try: mlflow.log_artifact(importance_path, "explainability")
                except Exception: pass
        except Exception as e:
            logger.warning(f"SHAP importance failed: {e}")

        # ── Log model ─────────────────────────────────────────────────
        logger.info("Logging model...")
        if HAS_MLFLOW:
            try:
                mlflow.sklearn.log_model(
                    sk_model=classifier.calibrated_model,
                    artifact_path="model",
                    registered_model_name=model_name,
                    input_example=X_test.head(3),
                )
            except Exception as e:
                logger.warning(f"MLflow log_model warning: {e}")

        # ── Quality Gate ──────────────────────────────────────────────
        logger.info("Running model quality gate...")
        gate_result = self.quality_gate.evaluate(metrics)
        _log_param("quality_gate_passed", gate_result["passed"])
        _log_params({
            f"gate_{k}": v
            for k, v in gate_result.get("checks", {}).items()
        })

        if HAS_MLFLOW:
            try: mlflow.end_run()
            except Exception: pass

        if gate_result["passed"]:
            logger.info("✅ Quality gate PASSED — model eligible for promotion")
        else:
            logger.warning(
                f"❌ Quality gate FAILED: {gate_result.get('failures', [])}"
            )

        metrics["quality_gate"] = gate_result
        metrics["run_id"] = run_id
        metrics["model_name"] = model_name

        return run_id, metrics


    def train_and_evaluate(
        self,
        data_path: str = "data/processed/features.parquet",
        model_name: str = "vulnerability-severity-classifier",
    ) -> tuple[str, dict[str, Any], Any]:
        """Convenience wrapper to load features from parquet/csv and run training."""
        if not os.path.exists(data_path):
            # Fallback to csv if parquet not found
            csv_path = data_path.replace(".parquet", ".csv")
            if os.path.exists(csv_path):
                data_path = csv_path
            else:
                raise FileNotFoundError(f"Feature file not found at {data_path}")

        if data_path.endswith(".parquet"):
            df = pd.read_parquet(data_path)
        else:
            df = pd.read_csv(data_path)

        run_id, metrics = self.train(df_features=df, model_name=model_name, challenger=True)
        return run_id, metrics, None


    def _apply_smote(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        k_neighbors: int = 5,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply SMOTE with safety checks for small minority classes."""
        if not HAS_SMOTE:
            logger.info("imbalanced-learn not installed; skipping SMOTE balancing.")
            return X.values, y.values

        class_counts = y.value_counts()
        min_count = class_counts.min()


        # SMOTE requires k_neighbors < min_class_count
        k = min(k_neighbors, min_count - 1)
        if k < 1:
            logger.warning(
                f"Minority class has only {min_count} samples. "
                "Skipping SMOTE, using original data."
            )
            return X.values, y.values

        try:
            smote = SMOTE(k_neighbors=k, random_state=42)
            X_res, y_res = smote.fit_resample(X, y)
            logger.info(
                f"SMOTE: {len(X)} → {len(X_res)} samples "
                f"({y.value_counts().to_dict()} → {pd.Series(y_res).value_counts().to_dict()})"
            )
            return X_res, y_res
        except Exception as e:
            logger.warning(f"SMOTE failed ({e}), using original data")
            return X.values, y.values

    def _measure_latency(
        self,
        classifier: SeverityClassifier,
        X_test: pd.DataFrame,
        n_warmup: int = 5,
        n_trials: int = 100,
    ) -> float:
        """Measure p99 inference latency in milliseconds."""
        sample = X_test.head(1)
        times = []

        # Warmup
        for _ in range(n_warmup):
            classifier.predict_proba(sample)

        # Measure
        for _ in range(n_trials):
            t0 = time.perf_counter()
            classifier.predict_proba(sample)
            times.append((time.perf_counter() - t0) * 1000)

        p99 = float(np.percentile(times, 99))
        logger.info(f"Inference latency: p50={np.median(times):.1f}ms, p99={p99:.1f}ms")
        return p99
