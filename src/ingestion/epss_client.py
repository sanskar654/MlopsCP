"""
EPSS API Client
===============
Client for the FIRST.org Exploit Prediction Scoring System (EPSS) API.

EPSS provides daily probability-of-exploitation scores (0-1) for all CVEs.
API endpoint: https://api.first.org/data/v1/epss

The score represents the probability that a CVE will be exploited in the
wild within the next 30 days, providing a critical signal beyond CVSS severity.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any, Generator

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

import csv
import gzip
import io

EPSS_BASE_URL = "https://api.first.org/data/v1/epss"
EPSS_BULK_CSV_URL = "https://epss.cyentia.com/epss_scores-current.csv.gz"
EPSS_BULK_CSV_FALLBACK_URL = "https://epss.empiricalsecurity.com/epss_scores-current.csv.gz"
EPSS_PAGE_SIZE = 10000  # API max


class EPSSClient:
    """
    Client for the FIRST EPSS API and daily bulk CSV releases.

    Usage:
        client = EPSSClient()

        # Bulk download daily scores
        for batch in client.fetch_bulk_csv():
            store_in_epss_history(batch)
    """

    def __init__(self, base_url: str = EPSS_BASE_URL, timeout: int = 120):
        self.base_url = base_url
        self.timeout = timeout

        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "VulnMLOps/1.0 (syngenta-evaluation)",
        })

    @retry(
        retry=retry_if_exception_type((requests.Timeout, requests.ConnectionError)),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _get(self, params: dict[str, Any]) -> dict[str, Any]:
        resp = self.session.get(self.base_url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def fetch_for_cves(
        self,
        cve_ids: list[str],
        score_date: date | None = None,
    ) -> list[dict[str, Any]]:
        """
        Fetch EPSS scores for a list of CVE IDs.
        Batches into groups of 100 (API limitation).

        Args:
            cve_ids: List of CVE ID strings
            score_date: Date for scores (defaults to latest)

        Returns:
            List of {'cve_id', 'score_date', 'epss_score', 'percentile'} dicts
        """
        if not cve_ids:
            return []

        results = []
        batch_size = 100  # EPSS API handles comma-separated, max ~100 reliable

        for i in range(0, len(cve_ids), batch_size):
            batch = cve_ids[i:i + batch_size]
            params: dict[str, Any] = {
                "cve": ",".join(batch),
                "limit": batch_size,
            }
            if score_date:
                params["date"] = score_date.isoformat()

            try:
                data = self._get(params)
                for item in data.get("data", []):
                    results.append(self._parse_entry(item))
                logger.debug(f"EPSS: fetched {len(batch)} CVE scores")
            except Exception as e:
                logger.error(f"EPSS fetch failed for batch starting at {i}: {e}")

        return results

    def fetch_all(
        self,
        score_date: date | None = None,
    ) -> Generator[list[dict[str, Any]], None, None]:
        """
        Fetch ALL CVE EPSS scores for a given date (full dataset).
        Yields batches for memory efficiency.

        Args:
            score_date: Date for scores (defaults to latest available)

        Yields:
            Batches of parsed EPSS score dicts
        """
        offset = 0
        total = None

        params: dict[str, Any] = {
            "limit": EPSS_PAGE_SIZE,
            "offset": offset,
            "order": "epss-desc",  # Highest risk first
        }
        if score_date:
            params["date"] = score_date.isoformat()

        while True:
            params["offset"] = offset
            try:
                data = self._get(params)
            except Exception as e:
                logger.error(f"EPSS bulk fetch failed at offset {offset}: {e}")
                raise

            if total is None:
                total = data.get("total", 0)
                logger.info(f"EPSS: fetching {total} total scores")

            entries = data.get("data", [])
            if not entries:
                break

            yield [self._parse_entry(e) for e in entries]
            offset += len(entries)

            if offset >= total:
                break

    def fetch_high_risk(
        self,
        min_score: float = 0.1,
        score_date: date | None = None,
    ) -> list[dict[str, Any]]:
        """
        Fetch only high-risk CVEs (EPSS score above threshold).
        Efficient for daily monitoring without full dataset download.
        """
        params: dict[str, Any] = {
            "epss-gt": min_score,
            "limit": EPSS_PAGE_SIZE,
            "order": "epss-desc",
        }
        if score_date:
            params["date"] = score_date.isoformat()

        results = []
        offset = 0
        total = None

        while True:
            params["offset"] = offset
            try:
                data = self._get(params)
            except Exception as e:
                logger.error(f"EPSS high-risk fetch failed: {e}")
                raise

            if total is None:
                total = data.get("total", 0)
                logger.info(f"EPSS: {total} CVEs with score > {min_score}")

            entries = data.get("data", [])
            if not entries:
                break

            results.extend([self._parse_entry(e) for e in entries])
            offset += len(entries)
            if offset >= total:
                break

        return results

    def _parse_entry(self, item: dict[str, Any]) -> dict[str, Any]:
        """Parse a single EPSS API response item."""
        return {
            "cve_id": item.get("cve", ""),
            "score_date": item.get("date", date.today().isoformat()),
            "epss_score": float(item.get("epss", 0.0)),
            "percentile": float(item.get("percentile", 0.0)),
        }

    def fetch_bulk_csv(
        self,
        batch_size: int = 10000,
        snapshot_date: date | None = None,
    ) -> Generator[list[dict[str, Any]], None, None]:
        """
        Download daily gzipped bulk EPSS CSV dataset and yield parsed batches.

        Args:
            batch_size: Number of records per batch yielded.
            snapshot_date: Date associated with score snapshot (defaults to today).

        Yields:
            Batches of dicts with keys: cve_id, score, percentile, snapshot_date.
        """
        today_str = (snapshot_date or date.today()).isoformat()
        urls_to_try = [EPSS_BULK_CSV_URL, EPSS_BULK_CSV_FALLBACK_URL]
        response = None

        for url in urls_to_try:
            try:
                logger.info(f"Downloading EPSS bulk CSV from {url}")
                resp = self.session.get(url, timeout=self.timeout, stream=True)
                resp.raise_for_status()
                response = resp
                break
            except Exception as e:
                logger.warning(f"Failed to download EPSS bulk CSV from {url}: {e}")

        if response is None:
            raise RuntimeError("All EPSS bulk CSV download sources failed.")

        # Decompress gzipped content line by line
        with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as gz_file:
            text_stream = io.TextIOWrapper(gz_file, encoding="utf-8")
            reader = csv.reader(text_stream)

            # Skip comment lines if present (EPSS CSV starts with #model_version)
            batch = []
            for row in reader:
                if not row or row[0].startswith("#") or row[0] == "cve":
                    continue
                if len(row) >= 3:
                    cve_id, epss_val, perc_val = row[0].strip(), row[1].strip(), row[2].strip()
                    try:
                        batch.append({
                            "cve_id": cve_id,
                            "score": float(epss_val),
                            "percentile": float(perc_val),
                            "snapshot_date": today_str,
                        })
                    except ValueError:
                        continue

                if len(batch) >= batch_size:
                    yield batch
                    batch = []

            if batch:
                yield batch

