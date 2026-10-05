"""
Asset Inventory & CPE Matcher
==============================
Manages simulated enterprise software assets and matches them against
NVD affected-product/CPE data to populate asset_cve_map.
"""

from __future__ import annotations

import logging
from typing import Any
import pandas as pd

logger = logging.getLogger(__name__)

# Simulated enterprise software asset inventory fixture
DEFAULT_ASSET_FIXTURES: list[dict[str, Any]] = [
    {
        "asset_id": "AST-FW-001",
        "asset_name": "Edge Perimeter Firewall",
        "product": "paloalto_networks_pan-os",
        "version": "10.1.0",
        "criticality": 5,
        "internet_facing": True,
        "asset_type": "network_device",
        "owner": "NetSecOps",
    },
    {
        "asset_id": "AST-APP-102",
        "asset_name": "Customer Portal Backend",
        "product": "log4j",
        "version": "2.14.1",
        "criticality": 5,
        "internet_facing": True,
        "asset_type": "container",
        "owner": "AppSec",
    },
    {
        "asset_id": "AST-APP-204",
        "asset_name": "Billing API Gateway",
        "product": "spring_framework",
        "version": "5.3.17",
        "criticality": 4,
        "internet_facing": True,
        "asset_type": "server",
        "owner": "FinanceIT",
    },
    {
        "asset_id": "AST-DB-001",
        "asset_name": "Production Customer DB Cluster",
        "product": "postgresql",
        "version": "13.2",
        "criticality": 5,
        "internet_facing": False,
        "asset_type": "server",
        "owner": "DBOps",
    },
    {
        "asset_id": "AST-WEB-050",
        "asset_name": "Internal Knowledge Base",
        "product": "nginx",
        "version": "1.18.0",
        "criticality": 2,
        "internet_facing": False,
        "asset_type": "server",
        "owner": "InternalIT",
    },
    {
        "asset_id": "AST-WKS-890",
        "asset_name": "Executive Workstation Suite",
        "product": "chrome",
        "version": "96.0.4664",
        "criticality": 3,
        "internet_facing": True,
        "asset_type": "workstation",
        "owner": "EndUserOps",
    },
    {
        "asset_id": "AST-APP-301",
        "asset_name": "Legacy Partner Integration Portal",
        "product": "struts",
        "version": "2.3.34",
        "criticality": 4,
        "internet_facing": True,
        "asset_type": "server",
        "owner": "AppSec",
    },
]


def match_assets_to_cves(
    df_cves: pd.DataFrame,
    df_assets: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Match asset inventory products against raw/parsed CVE CPE lists or descriptions.

    Args:
        df_cves: DataFrame of CVE records with columns ['cve_id', 'cpe_list', 'description', 'product']
        df_assets: Optional DataFrame of assets (defaults to DEFAULT_ASSET_FIXTURES)

    Returns:
        DataFrame with columns ['asset_id', 'cve_id', 'matched_at']
    """
    if df_assets is None or len(df_assets) == 0:
        df_assets = pd.DataFrame(DEFAULT_ASSET_FIXTURES)

    matches = []

    for _, asset in df_assets.iterrows():
        asset_id = asset["asset_id"]
        product_name = str(asset["product"]).lower()

        for _, cve in df_cves.iterrows():
            cve_id = cve["cve_id"]
            matched = False

            # Match against CPE list if available
            cpe_list = cve.get("cpe_list", [])
            if isinstance(cpe_list, (list, tuple)):
                for cpe in cpe_list:
                    if product_name in str(cpe).lower():
                        matched = True
                        break

            # Fallback match against cve product field or description keyword
            if not matched:
                cve_prod = str(cve.get("product", "")).lower()
                cve_desc = str(cve.get("description", "")).lower()
                if product_name in cve_prod or product_name in cve_desc:
                    matched = True

            if matched:
                matches.append({
                    "asset_id": asset_id,
                    "cve_id": cve_id,
                })

    df_map = pd.DataFrame(matches).drop_duplicates()
    logger.info(f"Asset-CVE Matcher: Found {len(df_map)} matches across {len(df_assets)} assets")
    return df_map
