"""
CISA KEV Client
===============
Client for the CISA Known Exploited Vulnerabilities (KEV) catalog.

The KEV catalog is a JSON file published by CISA at:
https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json

Updated frequently (typically weekdays). We detect changes via ETag/Last-Modified
headers to avoid unnecessary full downloads.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

CISA_KEV_URL = (
    "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"
)
ETAG_CACHE_FILE = Path("/tmp/cisa_kev_etag.txt")


class CISAKEVClient:
    """
    Client for the CISA KEV catalog.

    Usage:
        client = CISAKEVClient()
        entries, changed = client.fetch(cache_path="/tmp/kev_cache.json")
        if changed:
            process(entries)
    """

    def __init__(
        self,
        url: str = CISA_KEV_URL,
        timeout: int = 30,
    ):
        self.url = url
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "VulnMLOps/1.0 (syngenta-evaluation)",
        })

    @retry(
        retry=retry_if_exception_type((requests.Timeout, requests.ConnectionError)),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def fetch(
        self,
        cache_path: str | Path | None = None,
        force_refresh: bool = False,
    ) -> tuple[list[dict[str, Any]], bool]:
        """
        Fetch the KEV catalog. Returns (entries, changed).

        Uses ETag/If-None-Match to avoid re-downloading if unchanged.

        Args:
            cache_path: Optional path to cache the raw JSON locally
            force_refresh: Skip ETag check and always download

        Returns:
            Tuple of (list of KEV entry dicts, bool indicating new data)
        """
        headers = {}
        cached_etag = self._load_cached_etag()

        if cached_etag and not force_refresh:
            headers["If-None-Match"] = cached_etag

        resp = self.session.get(self.url, headers=headers, timeout=self.timeout)

        if resp.status_code == 304:
            logger.info("CISA KEV: No changes since last fetch (304 Not Modified)")
            # Load from cache if available
            if cache_path and Path(cache_path).exists():
                with open(cache_path) as f:
                    data = json.load(f)
                return self._parse_entries(data), False
            return [], False

        resp.raise_for_status()

        # Cache the ETag for future requests
        if etag := resp.headers.get("ETag"):
            self._save_etag(etag)

        data = resp.json()

        if cache_path:
            cache_path = Path(cache_path)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(f"CISA KEV cached to {cache_path}")

        entries = self._parse_entries(data)
        catalog_version = data.get("catalogVersion", "unknown")
        date_released = data.get("dateReleased", "unknown")

        logger.info(
            f"CISA KEV: {len(entries)} entries, "
            f"version={catalog_version}, released={date_released}"
        )
        return entries, True

    def fetch_as_set(self) -> set[str]:
        """Return just the set of CVE IDs in the KEV catalog (fast lookup)."""
        entries, _ = self.fetch()
        return {e["cve_id"] for e in entries}

    def fetch_catalog(
        self,
        cache_path: str | Path | None = None,
        force_refresh: bool = False,
    ) -> list[dict[str, Any]]:
        """Alias for fetch() returning entries list directly for pipeline DAGs."""
        entries, _ = self.fetch(cache_path=cache_path, force_refresh=force_refresh)
        return entries


    def _parse_entries(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        """Parse KEV JSON into normalized flat dicts."""
        raw_vulns = data.get("vulnerabilities", [])
        parsed = []
        for v in raw_vulns:
            parsed.append({
                "cve_id": v.get("cveID", ""),
                "vendor_project": v.get("vendorProject", ""),
                "product": v.get("product", ""),
                "vulnerability_name": v.get("vulnerabilityName", ""),
                "date_added": v.get("dateAdded", ""),
                "short_description": v.get("shortDescription", ""),
                "required_action": v.get("requiredAction", ""),
                "due_date": v.get("dueDate", ""),
                "known_ransomware_campaign_use": v.get("knownRansomwareCampaignUse", "Unknown"),
                "notes": v.get("notes", ""),
            })
        return parsed

    def _load_cached_etag(self) -> str | None:
        try:
            return ETAG_CACHE_FILE.read_text().strip() or None
        except FileNotFoundError:
            return None

    def _save_etag(self, etag: str) -> None:
        ETAG_CACHE_FILE.write_text(etag)


def get_kev_cve_ids(url: str = CISA_KEV_URL) -> set[str]:
    """Convenience function: return set of CVE IDs in CISA KEV."""
    client = CISAKEVClient(url=url)
    return client.fetch_as_set()
