"""
NVD API 2.0 Client
==================
Production-grade client for the National Vulnerability Database REST API v2.

Implements:
- Pagination with startIndex/resultsPerPage
- Exponential backoff and retry
- Rate limiting (5 req/30s without key, 50 req/30s with key)
- Incremental sync by lastModStartDate
- Schema-compliant response parsing
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Generator

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


NVD_BASE_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
PAGE_SIZE = 2000  # Max allowed by NVD API
# Rate limit windows
RATE_LIMIT_NO_KEY = (5, 30)    # 5 requests per 30 seconds
RATE_LIMIT_WITH_KEY = (50, 30) # 50 requests per 30 seconds


class NVDRateLimiter:
    """Token-bucket rate limiter for NVD API compliance."""

    def __init__(self, has_api_key: bool = False):
        if has_api_key:
            self.max_requests, self.window_seconds = RATE_LIMIT_WITH_KEY
        else:
            self.max_requests, self.window_seconds = RATE_LIMIT_NO_KEY
        self._requests: list[float] = []

    def wait_if_needed(self) -> None:
        """Block until a request slot is available."""
        now = time.monotonic()
        # Drop requests outside the current window
        self._requests = [t for t in self._requests if now - t < self.window_seconds]
        if len(self._requests) >= self.max_requests:
            sleep_time = self.window_seconds - (now - self._requests[0]) + 0.1
            logger.debug(f"Rate limit: sleeping {sleep_time:.2f}s")
            time.sleep(max(0, sleep_time))
        self._requests.append(time.monotonic())


class NVDClient:
    """
    Client for the NVD CVE API 2.0.

    Usage:
        client = NVDClient(api_key="your-key-here")
        for batch in client.fetch_cves_since(days=30):
            process(batch)
    """

    def __init__(
        self,
        api_key: str | None = None,
        timeout: int = 60,
        max_retries: int = 5,
    ):
        self.api_key = api_key
        self.timeout = timeout
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({"apiKey": api_key})
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "VulnMLOps/1.0 (syngenta-evaluation; contact@mlops.local)"
        })
        self.rate_limiter = NVDRateLimiter(has_api_key=bool(api_key))
        self.max_retries = max_retries

    @retry(
        retry=retry_if_exception_type((requests.Timeout, requests.ConnectionError, requests.HTTPError)),
        wait=wait_exponential(multiplier=3, min=6, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _get(self, params: dict[str, Any]) -> dict[str, Any]:
        """Execute a rate-limited GET request against the NVD API."""
        self.rate_limiter.wait_if_needed()
        resp = self.session.get(NVD_BASE_URL, params=params, timeout=self.timeout)

        if resp.status_code == 429:
            logger.warning("NVD API 429 Rate Limit hit. Backing off 6s...")
            time.sleep(6.0)
            resp.raise_for_status()
        if resp.status_code == 403:
            raise PermissionError("NVD API key invalid or rate limit exceeded (403)")
        if resp.status_code == 404:
            return {"totalResults": 0, "vulnerabilities": []}
        if resp.status_code == 503:
            raise requests.ConnectionError("NVD API service unavailable (503)")
        resp.raise_for_status()
        return resp.json()


    def fetch_cves_since(
        self,
        days: int = 30,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> Generator[list[dict[str, Any]], None, None]:
        """
        Yield batches of CVE records modified since the given date.

        Args:
            days: Number of days to look back (ignored if start_date set)
            start_date: Explicit start datetime (UTC)
            end_date: Explicit end datetime (UTC, defaults to now)

        Yields:
            List of raw CVE dicts from NVD API response
        """
        now = datetime.now(timezone.utc)
        if start_date is None:
            start_date = now - timedelta(days=days)
        if end_date is None:
            end_date = now

        # NVD API requires ISO 8601 format with T and timezone
        fmt = "%Y-%m-%dT%H:%M:%S.000"
        start_str = start_date.strftime(fmt)
        end_str = end_date.strftime(fmt)

        logger.info(
            f"Fetching CVEs modified between {start_str} and {end_str}"
        )

        start_index = 0
        total_results = None

        while True:
            params = {
                "lastModStartDate": start_str,
                "lastModEndDate": end_str,
                "startIndex": start_index,
                "resultsPerPage": PAGE_SIZE,
            }
            try:
                data = self._get(params)
            except Exception as e:
                logger.error(f"NVD API error at index {start_index}: {e}")
                raise

            if total_results is None:
                total_results = data.get("totalResults", 0)
                logger.info(f"Total CVEs to fetch: {total_results}")

            vulns = data.get("vulnerabilities", [])
            if not vulns:
                break

            yield vulns
            logger.info(
                f"Fetched {start_index + len(vulns)}/{total_results} CVEs"
            )

            start_index += len(vulns)
            if start_index >= total_results:
                break

    def fetch_cve_by_id(self, cve_id: str) -> dict[str, Any] | None:
        """Fetch a single CVE by ID."""
        params = {"cveId": cve_id}
        try:
            data = self._get(params)
            vulns = data.get("vulnerabilities", [])
            return vulns[0] if vulns else None
        except Exception as e:
            logger.error(f"Failed to fetch {cve_id}: {e}")
            return None

    def fetch_all_cves(
        self,
        start_year: int = 2020,
    ) -> Generator[list[dict[str, Any]], None, None]:
        """
        Fetch all CVEs from a given start year to present.
        Processes year-by-year to handle large datasets.
        """
        current_year = datetime.now(timezone.utc).year
        for year in range(start_year, current_year + 1):
            start = datetime(year, 1, 1, tzinfo=timezone.utc)
            end = datetime(min(year + 1, current_year + 1), 1, 1, tzinfo=timezone.utc)
            end = min(end, datetime.now(timezone.utc))
            logger.info(f"Fetching CVEs for year {year}")
            yield from self.fetch_cves_since(start_date=start, end_date=end)


def parse_nvd_cve(raw_cve: dict[str, Any]) -> dict[str, Any] | None:
    """
    Parse a raw NVD API 2.0 CVE record into a flat dict for storage.

    Handles missing CVSS, multiple metric versions (v3.1 preferred over v3.0).
    Returns None if the record is invalid.
    """
    try:
        cve = raw_cve.get("cve", {})
        cve_id = cve.get("id")
        if not cve_id:
            return None

        # Description (English preferred)
        descriptions = cve.get("descriptions", [])
        description = ""
        for d in descriptions:
            if d.get("lang") == "en":
                description = d.get("value", "")
                break

        # Publication dates
        published = cve.get("published", "")
        last_modified = cve.get("lastModified", "")

        # CVSS Metrics — prefer v3.1 > v3.0 > v2.0
        metrics = cve.get("metrics", {})
        cvss_data = _extract_cvss_v3(metrics)
        cvss_v2_score = _extract_cvss_v2_score(metrics)

        # CWE
        weaknesses = cve.get("weaknesses", [])
        cwe_ids = []
        for w in weaknesses:
            for desc in w.get("description", []):
                if desc.get("lang") == "en" and desc.get("value", "").startswith("CWE-"):
                    cwe_ids.append(desc["value"])

        # References
        references = cve.get("references", [])
        ref_urls = [r.get("url", "") for r in references if r.get("url")]

        # CPE (affected products)
        configurations = cve.get("configurations", [])
        cpe_list = _extract_cpes(configurations)

        return {
            "cve_id": cve_id,
            "published_date": published,
            "last_modified_date": last_modified,
            "vuln_status": cve.get("vulnStatus"),
            "description": description,
            # CVSS v3
            "cvss_version": cvss_data.get("version"),
            "cvss_vector_string": cvss_data.get("vectorString"),
            "attack_vector": cvss_data.get("attackVector"),
            "attack_complexity": cvss_data.get("attackComplexity"),
            "privileges_required": cvss_data.get("privilegesRequired"),
            "user_interaction": cvss_data.get("userInteraction"),
            "scope": cvss_data.get("scope"),
            "confidentiality_impact": cvss_data.get("confidentialityImpact"),
            "integrity_impact": cvss_data.get("integrityImpact"),
            "availability_impact": cvss_data.get("availabilityImpact"),
            "base_score": cvss_data.get("baseScore"),
            "base_severity": cvss_data.get("baseSeverity"),
            "exploitability_score": cvss_data.get("exploitabilityScore"),
            "impact_score": cvss_data.get("impactScore"),
            # CVSS v2 fallback
            "cvss_v2_score": cvss_v2_score.get("baseScore"),
            "cvss_v2_severity": cvss_v2_score.get("baseSeverity"),
            # Metadata
            "cwe_ids": cwe_ids,
            "reference_urls": ref_urls,
            "cpe_list": cpe_list,
            "vendor_count": len(cpe_list),
            "reference_count": len(ref_urls),
        }
    except Exception as e:
        logger.warning(f"Failed to parse CVE record: {e}")
        return None


def _extract_cvss_v3(metrics: dict) -> dict:
    """Extract CVSS v3.1 (preferred) or v3.0 metrics."""
    for key in ("cvssMetricV31", "cvssMetricV30"):
        entries = metrics.get(key, [])
        if entries:
            # Prefer primary source
            for entry in entries:
                if entry.get("type") == "Primary":
                    cvss = entry.get("cvssData", {})
                    return {
                        **cvss,
                        "exploitabilityScore": entry.get("exploitabilityScore"),
                        "impactScore": entry.get("impactScore"),
                    }
            # Fallback to first available
            cvss = entries[0].get("cvssData", {})
            return {
                **cvss,
                "exploitabilityScore": entries[0].get("exploitabilityScore"),
                "impactScore": entries[0].get("impactScore"),
            }
    return {}


def _extract_cvss_v2_score(metrics: dict) -> dict:
    """Extract CVSS v2 base score as fallback."""
    entries = metrics.get("cvssMetricV2", [])
    if entries:
        cvss = entries[0].get("cvssData", {})
        return {
            "baseScore": cvss.get("baseScore"),
            "baseSeverity": entries[0].get("baseSeverity"),
        }
    return {}


def _extract_cpes(configurations: list[dict]) -> list[str]:
    """Recursively extract all CPE strings from configurations."""
    cpes = []
    for config in configurations:
        for node in config.get("nodes", []):
            for cpe_match in node.get("cpeMatch", []):
                if cpe := cpe_match.get("criteria"):
                    cpes.append(cpe)
    return cpes
