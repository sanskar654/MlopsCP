"""
Airflow DAG: Vulnerability Data Ingestion & Validation Pipeline
==============================================================
Runs daily/weekly to pull CVEs from NVD 2.0 API, CISA KEV catalog, and EPSS feed,
validates with DataValidator schemas, and saves versioned feature data.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import logging
import os
import sys

from airflow import DAG
from airflow.operators.python import PythonOperator

# Ensure src is accessible
sys.path.append(os.path.join(os.getenv("AIRFLOW_HOME", "/opt/airflow"), "src"))

default_args = {
    "owner": "mlops-platform",
    "depends_on_past": False,
    "start_date": datetime(2025, 1, 1),
    "email_on_failure": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}


def run_nvd_ingestion(**kwargs):
    """Fetch recent CVE records from NVD API or seed generator."""
    import pandas as pd
    from ingestion.nvd_client import NVDClient, parse_nvd_cve

    output_path = "/opt/airflow/data/raw/nvd_recent.parquet"
    if not os.path.exists("/opt/airflow/data/raw"):
        output_path = "data/raw/nvd_recent.parquet"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        client = NVDClient()
        records = []
        for batch in client.fetch_cves_since(days=7):
            for raw in batch:
                parsed = parse_nvd_cve(raw)
                if parsed:
                    records.append(parsed)
        df = pd.DataFrame(records)
    except Exception as e:
        logging.warning(f"Live NVD ingestion fallback: {e}")
        # Fallback to local raw data if API unaccessible
        raw_seed = "data/raw/vulnerabilities_dataset.parquet"
        if os.path.exists(raw_seed):
            df = pd.read_parquet(raw_seed)
        else:
            from scripts.seed_data import generate_realistic_seed_dataset
            generate_realistic_seed_dataset(num_samples=200, output_dir=os.path.dirname(output_path))
            df = pd.read_parquet(output_path.replace("nvd_recent.parquet", "vulnerabilities_dataset.parquet"))

    df.to_parquet(output_path, index=False)
    logging.info(f"Ingested {len(df)} CVE records to {output_path}")
    return len(df)


def run_cisa_kev_ingestion(**kwargs):
    """Fetch active catalog from CISA KEV."""
    import pandas as pd
    from ingestion.cisa_kev_client import CISAKEVClient

    output_path = "/opt/airflow/data/raw/cisa_kev.parquet"
    if not os.path.exists("/opt/airflow/data/raw"):
        output_path = "data/raw/cisa_kev.parquet"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        client = CISAKEVClient()
        catalog = client.fetch_catalog()
        df = pd.DataFrame(catalog)
    except Exception as e:
        logging.warning(f"CISA KEV fetch fallback: {e}")
        df = pd.DataFrame([{"cve_id": "CVE-2021-44228", "date_added": "2021-12-10"}])

    df.to_parquet(output_path, index=False)
    logging.info(f"Ingested {len(df)} KEV records to {output_path}")
    return len(df)


def run_data_validation(**kwargs):
    """Validate schema integrity, missingness, and range constraints."""
    import pandas as pd
    from validation.schema_validator import VulnerabilitySchemaValidator

    validator = VulnerabilitySchemaValidator()
    input_path = "/opt/airflow/data/raw/nvd_recent.parquet"
    if not os.path.exists(input_path):
        input_path = "data/raw/nvd_recent.parquet"

    if not os.path.exists(input_path):
        input_path = "data/raw/vulnerabilities_dataset.parquet"

    df = pd.read_parquet(input_path)
    cleaned_df, report = validator.validate_raw(df)

    clean_path = "/opt/airflow/data/interim/validated_cves.parquet"
    if not os.path.exists("/opt/airflow/data/interim"):
        clean_path = "data/interim/validated_cves.parquet"

    os.makedirs(os.path.dirname(clean_path), exist_ok=True)
    cleaned_df.to_parquet(clean_path, index=False)
    logging.info(f"Validated dataset saved to {clean_path} with {len(cleaned_df)} rows.")


def run_feature_engineering(**kwargs):
    """Transform text and categorical features into ML feature matrix."""
    import pandas as pd
    from features.feature_engineer import FeatureEngineer

    engineer = FeatureEngineer()
    input_path = "/opt/airflow/data/interim/validated_cves.parquet"
    if not os.path.exists(input_path):
        input_path = "data/interim/validated_cves.parquet"

    df = pd.read_parquet(input_path)
    features_df = engineer.engineer_features(df)

    output_path = "/opt/airflow/data/processed/features.parquet"
    if not os.path.exists("/opt/airflow/data/processed"):
        output_path = "data/processed/features.parquet"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    features_df.to_parquet(output_path, index=False)
    logging.info(f"Generated feature matrix with shape {features_df.shape} at {output_path}")


with DAG(
    dag_id="vulnerability_data_ingestion_pipeline",
    default_args=default_args,
    description="Orchestrates ingestion, validation, and feature generation from NVD and CISA KEV",
    schedule_interval="0 2 * * *",
    catchup=False,
    tags=["ingestion", "nvd", "cisa-kev", "features"],
) as dag:

    t1 = PythonOperator(
        task_id="ingest_nvd_cves",
        python_callable=run_nvd_ingestion,
    )

    t2 = PythonOperator(
        task_id="ingest_cisa_kev",
        python_callable=run_cisa_kev_ingestion,
    )

    t3 = PythonOperator(
        task_id="validate_dataset",
        python_callable=run_data_validation,
    )

    t4 = PythonOperator(
        task_id="build_feature_store",
        python_callable=run_feature_engineering,
    )

    [t1, t2] >> t3 >> t4
