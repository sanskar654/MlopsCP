"""
Airflow DAG: Drift Detection & Automated Retraining Trigger
==========================================================
Runs periodically to compare production inference logs against baseline training data.
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
    "email": [os.getenv("ALERT_EMAIL", "mlops-alerts@syngenta-sec.org")],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def check_production_drift(**kwargs):
    """Run statistical and PSI drift analysis between baseline and recent inference."""
    import pandas as pd
    from monitoring.drift_detector import DriftMonitor
    from monitoring.alerting import AlertManager

    ref_path = "/opt/airflow/data/processed/features.parquet"
    curr_path = "/opt/airflow/data/inference_logs/recent_cves.parquet"

    if not os.path.exists(curr_path):
        logging.info("No new inference logs to test for drift. Skipping.")
        return "no_drift_detected"

    monitor = DriftMonitor(reference_data_path=ref_path, drift_share_threshold=0.25)
    curr_df = pd.read_parquet(curr_path)
    report = monitor.run_drift_analysis(curr_df)

    monitor.save_report(report, "/opt/airflow/data/monitoring/latest_drift_report.json")

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
    description="Monitors production feature/prediction drift and triggers adaptive retraining",
    schedule_interval="0 */6 * * *",  # Every 6 hours
    catchup=False,
    tags=["monitoring", "drift", "evidently", "adaptive"],
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
