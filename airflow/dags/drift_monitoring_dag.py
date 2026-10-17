"""
Airflow DAG: Drift Detection & Automated Retraining Trigger
==========================================================
Runs periodically to compare production inference logs against baseline training data.
Reads the last 7 days from `predictions_log` in PostgreSQL.
Monitors:
- Tabular feature distribution drift (KS-Test & PSI)
- Text embedding drift (TF-IDF semantic centroid shift)
- CWE category distribution drift (Chi-Square)

If dataset drift exceeds threshold, triggers the retraining DAG automatically.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import logging
import os
import sys

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

sys.path.append(os.path.join(os.getenv("AIRFLOW_HOME", "/opt/airflow"), "src"))

default_args = {
    "owner": "mlops-platform",
    "depends_on_past": False,
    "start_date": datetime(2025, 1, 1),
    "email_on_failure": True,
    "email": [os.getenv("ALERT_EMAIL", "mlops-alerts@vulnerability-mlops.local")],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def check_production_drift(**kwargs):
    """
    Run statistical, text embedding, and CWE distribution drift analysis
    by querying the last 7 days from PostgreSQL predictions_log.
    """
    import pandas as pd
    from monitoring.drift_detector import DriftMonitor
    from monitoring.alerting import AlertManager

    base_dir = os.getenv("AIRFLOW_HOME", "/opt/airflow")
    ref_path = os.path.join(base_dir, "data/processed/features.parquet")
    if not os.path.exists(ref_path):
        ref_path = "data/processed/features.parquet"

    monitor = DriftMonitor(reference_data_path=ref_path, drift_share_threshold=0.25)

    # Read last 7 days of inference from PostgreSQL predictions_log
    curr_df = monitor.load_recent_production_logs(days=7)

    if curr_df.empty:
        logging.info("No inference logs found in predictions_log for the last 7 days. Skipping.")
        return "no_drift_detected"

    logging.info("Analyzing drift on %d recent production predictions...", len(curr_df))
    report = monitor.run_drift_analysis(curr_df)

    report_path = os.path.join(base_dir, "data/monitoring/latest_drift_report.json")
    monitor.save_report(report, report_path)

    alerter = AlertManager()
    alerter.check_drift_alert(report)

    if report.get("dataset_drift", False):
        logging.warning("Significant data drift detected! Branching to trigger retraining.")
        return "trigger_retraining_dag"
    else:
        logging.info("Data distribution is stable. No retraining necessary.")
        return "no_drift_detected"


def drift_stable_log(**kwargs):
    logging.info("Drift check passed: model remains stable.")


with DAG(
    dag_id="vulnerability_drift_monitoring_pipeline",
    default_args=default_args,
    description="Monitors production feature, text embedding, and CWE drift from PostgreSQL logs",
    schedule_interval="0 */6 * * *",  # Every 6 hours
    catchup=False,
    tags=["monitoring", "drift", "text_drift", "cwe_drift", "adaptive"],
) as dag:

    drift_branch = BranchPythonOperator(
        task_id="evaluate_drift",
        python_callable=check_production_drift,
    )

    stable_task = PythonOperator(
        task_id="no_drift_detected",
        python_callable=drift_stable_log,
    )

    trigger_retrain = TriggerDagRunOperator(
        task_id="trigger_retraining_dag",
        trigger_dag_id="vulnerability_model_training_pipeline",
        wait_for_completion=False,
    )

    drift_branch >> [stable_task, trigger_retrain]
