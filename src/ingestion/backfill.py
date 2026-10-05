"""
Real Data Backfill & Merging Script
====================================
Fetches and merges real data directly from official vulnerability APIs:
1. CISA KEV (Known Exploited Vulnerabilities Catalog)
2. FIRST EPSS (Exploit Prediction Scoring System API)
3. NVD API 2.0 (National Vulnerability Database)

Exports clean, leakage-free dataset to `data/processed/real_cve_dataset.parquet`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.ingestion.cisa_kev_client import CISAKEVClient
from src.ingestion.epss_client import EPSSClient
from src.ingestion.nvd_client import NVDClient, parse_nvd_cve
from src.features.cvss_parser import parse_cvss_vector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ingestion.backfill")


def fetch_and_merge_real_data(
    nvd_sample_size: int = 50,
    output_dir: str | Path = "data/processed"
) -> pd.DataFrame:
    """
    Fetch real data from CISA KEV, EPSS API, and NVD API, merge them into a unified dataset.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Fetch CISA KEV Catalog
    logger.info("Fetching CISA KEV Catalog...")
    kev_client = CISAKEVClient()
    kev_entries, _ = kev_client.fetch(force_refresh=True)
    logger.info(f"Retrieved {len(kev_entries)} real KEV records from CISA.")
    
    kev_df = pd.DataFrame(kev_entries)
    kev_cves = set(kev_df["cve_id"].unique()) if "cve_id" in kev_df.columns else set()

    # 2. Extract sample CVE IDs for NVD & EPSS query
    logger.info("Building target CVE ID list...")
    target_cve_ids = list(kev_cves)[:nvd_sample_size] if kev_cves else []
    
    # Add key historical CVEs if not present
    key_cves = [
        "CVE-2021-44228", "CVE-2023-23397", "CVE-2024-21413", "CVE-2017-5638",
        "CVE-2020-1472", "CVE-2021-34527", "CVE-2022-22965", "CVE-2023-34362",
        "CVE-2024-30078", "CVE-2023-4863", "CVE-2023-38606", "CVE-2024-21338"
    ]
    for cve in key_cves:
        if cve not in target_cve_ids:
            target_cve_ids.append(cve)

    # 3. Fetch EPSS Scores
    logger.info("Fetching live EPSS scores from FIRST API...")
    epss_client = EPSSClient()
    epss_records = epss_client.fetch_for_cves(target_cve_ids)
    epss_df = pd.DataFrame(epss_records)
    logger.info(f"Retrieved {len(epss_df)} EPSS scores.")

    epss_map = dict(zip(epss_df["cve_id"], epss_df["epss_score"])) if not epss_df.empty else {}
    percentile_map = dict(zip(epss_df["cve_id"], epss_df["percentile"])) if not epss_df.empty else {}

    # 4. Fetch NVD CVE details
    logger.info(f"Fetching {len(target_cve_ids)} CVE records from NVD API 2.0...")
    nvd_client = NVDClient()
    parsed_records = []

    for i, cve_id in enumerate(target_cve_ids):
        if i > 0 and i % 10 == 0:
            logger.info(f"Progress: {i}/{len(target_cve_ids)} CVEs fetched from NVD API...")
        
        raw_cve = nvd_client.fetch_cve_by_id(cve_id)
        if raw_cve:
            parsed = parse_nvd_cve(raw_cve)
            if parsed:
                cve_code = parsed["cve_id"]
                parsed["in_cisa_kev"] = 1 if cve_code in kev_cves else 0
                parsed["epss_score"] = float(epss_map.get(cve_code, 0.01))
                parsed["epss_percentile"] = float(percentile_map.get(cve_code, 0.05))

                # Parse CVSS submetrics
                cvss_vec = parsed.get("cvss_vector", "")
                submetrics = parse_cvss_vector(cvss_vec)
                parsed.update({f"cvss_{k}": v for k, v in submetrics.items()})

                parsed_records.append(parsed)

    df = pd.DataFrame(parsed_records)
    logger.info(f"Successfully processed {len(df)} real CVE records.")

    # Save outputs
    parquet_path = output_dir / "real_cve_dataset.parquet"
    csv_path = output_dir / "real_cve_dataset.csv"
    
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved real dataset to {parquet_path} and {csv_path}")

    return df


if __name__ == "__main__":
    fetch_and_merge_real_data(nvd_sample_size=30)
