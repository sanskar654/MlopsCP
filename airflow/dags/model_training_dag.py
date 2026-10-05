"""
Airflow DAG: Model Retraining, Evaluation & Promotion Pipeline
=============================================================
Orchestrates temporal cross-validation, SMOTE balancing, XGBoost training,
SHAP explainability computation, quality gate verification, and MLflow registry promotion.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import logging
import os
import sys

from airflow import DAG
from airflow.operators.python import PythonOperator

sys.path.append(os.path.join(os.getenv("AIRFLOW_HOME", "/opt/airflow"), "src"))

default_args = {
    "owner": "mlops-platform",
    "depends_on_past": False,
    "start_date": datetime(2025, 1, 1),
    "email_on_failure": True,
    "email": [os.getenv("ALERT_EMAIL", "mlops-alerts@syngenta-sec.org")],
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}


def run_training_pipeline(**kwargs):
    """Executes MLflow-tracked model training and temporal validation."""
    import pandas as pd
    from training.trainer import VulnerabilityTrainer

    data_path = "/opt/airflow/data/processed/features.parquet"
    if not os.path.exists(data_path):
        data_path = "data/processed/features.parquet"

    trainer = VulnerabilityTrainer(
        experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME", "vulnerability_severity_v2"),
        tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"),
    )

    run_id, metrics, model = trainer.train_and_evaluate(data_path=data_path)
    kwargs["ti"].xcom_push(key="challenger_run_id", value=run_id)
    kwargs["ti"].xcom_push(key="challenger_metrics", value=metrics)
    logging.info(f"Training run completed: {run_id} with metrics: {metrics}")


def run_quality_gate_and_promotion(**kwargs):
    """Enforces quality gates before promoting challenger to Champion in MLflow."""
    from training.quality_gate import ModelQualityGate

    run_id = kwargs["ti"].xcom_pull(key="challenger_run_id", task_ids="train_challenger_model")
    metrics = kwargs["ti"].xcom_pull(key="challenger_metrics", task_ids="train_challenger_model")

    gate = ModelQualityGate(
        min_macro_f1=0.82,
        min_critical_recall=0.88,
        max_ece=0.10,
        max_latency_p95_ms=50.0,
    )

    passed, report = gate.evaluate(metrics, run_id=run_id)
    logging.info(f"Quality gate verdict: passed={passed}, report={report}")

    if passed:
        gate.promote_to_production(run_id=run_id, model_name="vulnerability_severity_classifier")
        logging.info(f"Challenger model {run_id} promoted to Champion in MLflow registry.")
    else:
        logging.warning(f"Challenger model {run_id} rejected by Quality Gate. Champion preserved.")


with DAG(
    dag_id="vulnerability_model_training_pipeline",
    default_args=default_args,
    description="Automated model retraining, temporal CV, calibration, and quality gate promotion",
    schedule_interval="0 4 * * 0",  # Weekly on Sunday at 4:00 AM UTC
    catchup=False,
    tags=["training", "mlflow", "xgboost", "quality-gate"],
) as dag:

    t1 = PythonOperator(
        task_id="train_challenger_model",
        python_callable=run_training_pipeline,
    )

    t2 = PythonOperator(
        task_id="evaluate_quality_gate_and_promote",
        python_callable=run_quality_gate_and_promotion,
    )

    t1 >> t2
