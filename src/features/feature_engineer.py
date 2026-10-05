"""
Feature Engineering Pipeline
==============================
Transforms raw vulnerability records into ML-ready features.

Design principles:
- NO data leakage: base_score and base_severity are NEVER features
- Temporal features use only data available at time of CVE publication
- Exploitation signals (KEV, EPSS) are looked up contemporaneously
- All features are documented and versioned
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from .cvss_parser import (
    cvss_submetrics_to_onehot,
    parse_cvss_vector,
    severity_to_numeric,
)

logger = logging.getLogger(__name__)

FEATURE_VERSION = "1.0"

# ─── Keyword patterns for text features ──────────────────────────────────────
KEYWORD_PATTERNS = {
    "has_rce_keyword": re.compile(
        r"\b(remote.?code.?exec|rce|arbitrary.?code|command.?exec|code.?injection)\b",
        re.IGNORECASE,
    ),
    "has_sqli_keyword": re.compile(
        r"\b(sql.?inject|sqli|database.?inject)\b",
        re.IGNORECASE,
    ),
    "has_xss_keyword": re.compile(
        r"\b(cross.?site.?script|xss|stored.?xss|reflected.?xss)\b",
        re.IGNORECASE,
    ),
    "has_overflow_keyword": re.compile(
        r"\b(buffer.?overflow|stack.?overflow|heap.?overflow|integer.?overflow|memory.?corrupt)\b",
        re.IGNORECASE,
    ),
    "has_auth_bypass_keyword": re.compile(
        r"\b(auth.?bypass|authentication.?bypass|bypass.?auth|unauthenticated|improper.?auth)\b",
        re.IGNORECASE,
    ),
    "has_priv_esc_keyword": re.compile(
        r"\b(privilege.?escal|priv.?esc|local.?priv|gain.?privilege|elevat.?privilege)\b",
        re.IGNORECASE,
    ),
    "has_dos_keyword": re.compile(
        r"\b(denial.?of.?service|dos|infinite.?loop|crash|hung|hang|exhaust)\b",
        re.IGNORECASE,
    ),
}

# ─── CWE category groupings ───────────────────────────────────────────────────
MEMORY_SAFETY_CWES = {
    "CWE-119", "CWE-120", "CWE-121", "CWE-122", "CWE-125",
    "CWE-126", "CWE-127", "CWE-415", "CWE-416", "CWE-787",
    "CWE-788", "CWE-824", "CWE-476",
}

INJECTION_CWES = {
    "CWE-77", "CWE-78", "CWE-79", "CWE-89", "CWE-90",
    "CWE-91", "CWE-94", "CWE-95", "CWE-96", "CWE-113",
    "CWE-917", "CWE-943",
}

AUTH_CWES = {
    "CWE-287", "CWE-288", "CWE-290", "CWE-294", "CWE-295",
    "CWE-306", "CWE-307", "CWE-384", "CWE-522", "CWE-798",
}

CONFIG_CWES = {
    "CWE-16", "CWE-183", "CWE-185", "CWE-200", "CWE-209",
    "CWE-250", "CWE-276", "CWE-732", "CWE-749",
}


class FeatureEngineer:
    """
    Transforms raw vulnerability data into ML feature vectors.

    Usage:
        fe = FeatureEngineer()
        df_features = fe.transform(df_raw, kev_cve_ids, epss_scores)
    """

    def __init__(self, reference_date: datetime | None = None):
        """
        Args:
            reference_date: The "current" date for temporal features.
                           Defaults to today. OVERRIDE for temporal CV splits
                           to prevent leakage.
        """
        self.reference_date = reference_date or datetime.now(timezone.utc)

    def transform(
        self,
        df_raw: pd.DataFrame,
        kev_cve_ids: set[str] | None = None,
        epss_scores: dict[str, dict[str, float]] | None = None,
    ) -> pd.DataFrame:
        """
        Transform raw vulnerability DataFrame into feature DataFrame.

        Args:
            df_raw: DataFrame from raw_vulnerabilities table
            kev_cve_ids: Set of CVE IDs in CISA KEV catalog
            epss_scores: Dict {cve_id: {"epss_score": float, "percentile": float}}

        Returns:
            DataFrame with all engineered features + severity_label + cve_id
        """
        kev_cve_ids = kev_cve_ids or set()
        epss_scores = epss_scores or {}

        records = []
        for _, row in df_raw.iterrows():
            try:
                record = self._engineer_single(
                    row.to_dict(), kev_cve_ids, epss_scores
                )
                if record is not None:
                    records.append(record)
            except Exception as e:
                logger.warning(f"Feature engineering failed for {row.get('cve_id')}: {e}")

        df = pd.DataFrame(records)
        logger.info(f"Engineered features for {len(df)} vulnerabilities")
        return df

    def engineer_features(
        self,
        df_raw: pd.DataFrame,
        kev_cve_ids: set[str] | None = None,
        epss_scores: dict[str, dict[str, float]] | None = None,
    ) -> pd.DataFrame:
        """Alias for transform."""
        return self.transform(df_raw, kev_cve_ids=kev_cve_ids, epss_scores=epss_scores)

    def fit_transform(self, df_raw: pd.DataFrame, y=None) -> pd.DataFrame:
        """Alias for scikit-learn pipeline compliance, returning pure Stage 1 feature matrix."""
        df = self.transform(df_raw)
        cols_to_drop = [c for c in ["in_cisa_kev", "epss_score", "epss_percentile", "cvss_score", "cvss_severity", "severity"] if c in df.columns]
        return df.drop(columns=cols_to_drop)


    def _engineer_single(

        self,
        row: dict[str, Any],
        kev_cve_ids: set[str],
        epss_scores: dict[str, dict[str, float]],
    ) -> dict[str, Any] | None:
        """Engineer features for a single CVE record."""
        cve_id = row.get("cve_id", "")
        if not cve_id:
            return None

        features: dict[str, Any] = {"cve_id": cve_id, "feature_version": FEATURE_VERSION}

        # ── CVSS Submetric Features (one-hot) ──────────────────────────────
        vector_string = row.get("cvss_vector_string")
        if not vector_string:
            # Try to reconstruct from stored submetrics
            vector_string = self._reconstruct_vector(row)

        parsed = parse_cvss_vector(vector_string)
        onehot = cvss_submetrics_to_onehot(parsed)
        features.update(onehot)

        # ── Temporal Features ───────────────────────────────────────────────
        published = self._parse_dt(row.get("published_date"))
        if published:
            delta = self.reference_date - published.replace(tzinfo=timezone.utc) \
                if published.tzinfo is None else self.reference_date - published
            features["days_since_published"] = max(0, delta.days)
            features["quarter_published"] = (published.month - 1) // 3 + 1
            features["year_published"] = published.year
            features["is_recent"] = delta.days < 90
        else:
            features["days_since_published"] = None
            features["quarter_published"] = None
            features["year_published"] = None
            features["is_recent"] = False

        # ── Exploitation Signals ────────────────────────────────────────────
        in_kev = cve_id in kev_cve_ids
        features["in_cisa_kev"] = in_kev

        epss = epss_scores.get(cve_id, {})
        features["epss_score"] = epss.get("epss_score", 0.0)
        features["epss_percentile"] = epss.get("percentile", 0.0)

        # Check for exploit references in URLs
        ref_urls = row.get("reference_urls") or []
        features["has_exploit_reference"] = any(
            any(kw in url.lower() for kw in
                ["exploit", "poc", "metasploit", "github.com", "exploit-db"])
            for url in ref_urls
        )

        # ── Text / NLP Features ─────────────────────────────────────────────
        desc = row.get("description") or ""
        features["description_length"] = len(desc)
        features["description_word_count"] = len(desc.split())

        for feat_name, pattern in KEYWORD_PATTERNS.items():
            features[feat_name] = bool(pattern.search(desc))

        # ── CWE Features ────────────────────────────────────────────────────
        cwe_ids = set(row.get("cwe_ids") or [])
        features["is_memory_safety_cwe"] = bool(cwe_ids & MEMORY_SAFETY_CWES)
        features["is_injection_cwe"] = bool(cwe_ids & INJECTION_CWES)
        features["is_auth_cwe"] = bool(cwe_ids & AUTH_CWES)
        features["is_config_cwe"] = bool(cwe_ids & CONFIG_CWES)

        # Numeric CWE category from first CWE
        features["cwe_category_id"] = self._extract_cwe_num(cwe_ids)

        # ── Reference / Metadata Features ───────────────────────────────────
        features["vendor_count"] = row.get("vendor_count") or 0
        features["reference_count"] = row.get("reference_count") or 0
        ref_urls_lower = [u.lower() for u in ref_urls]
        features["has_patch_reference"] = any(
            any(kw in u for kw in ["patch", "advisory", "update", "fix", "security"])
            for u in ref_urls_lower
        )

        # ── Target Label (NOT a feature — stored alongside for training) ─────
        severity = row.get("base_severity")
        if not severity and row.get("base_score"):
            severity = self._score_to_severity(float(row["base_score"]))
        features["severity_label"] = (severity or "").upper() or None
        features["severity_numeric"] = severity_to_numeric(features["severity_label"])

        return features

    def _reconstruct_vector(self, row: dict) -> str | None:
        """Try to reconstruct CVSS vector from stored submetric columns."""
        # Reverse map for reconstruction
        av_map = {"NETWORK": "N", "ADJACENT": "A", "LOCAL": "L", "PHYSICAL": "P"}
        ac_map = {"LOW": "L", "HIGH": "H"}
        pr_map = {"NONE": "N", "LOW": "L", "HIGH": "H"}
        ui_map = {"NONE": "N", "REQUIRED": "R"}
        s_map = {"UNCHANGED": "U", "CHANGED": "C"}
        i_map = {"NONE": "N", "LOW": "L", "HIGH": "H"}

        try:
            av = av_map.get(str(row.get("attack_vector", "")), "")
            ac = ac_map.get(str(row.get("attack_complexity", "")), "")
            pr = pr_map.get(str(row.get("privileges_required", "")), "")
            ui = ui_map.get(str(row.get("user_interaction", "")), "")
            s = s_map.get(str(row.get("scope", "")), "")
            c = i_map.get(str(row.get("confidentiality_impact", "")), "")
            i = i_map.get(str(row.get("integrity_impact", "")), "")
            a = i_map.get(str(row.get("availability_impact", "")), "")

            if all([av, ac, pr, ui, s, c, i, a]):
                v = row.get("cvss_version", "3.1") or "3.1"
                return f"CVSS:{v}/AV:{av}/AC:{ac}/PR:{pr}/UI:{ui}/S:{s}/C:{c}/I:{i}/A:{a}"
        except Exception:
            pass
        return None

    def _parse_dt(self, dt_val: Any) -> datetime | None:
        if isinstance(dt_val, datetime):
            return dt_val
        if isinstance(dt_val, str) and dt_val:
            try:
                # NVD format: "2023-11-14T16:15:27.780"
                dt_val = dt_val.split(".")[0].replace("Z", "")
                return datetime.fromisoformat(dt_val)
            except ValueError:
                pass
        return None

    def _extract_cwe_num(self, cwe_ids: set[str]) -> int | None:
        for cwe in cwe_ids:
            try:
                return int(cwe.replace("CWE-", ""))
            except ValueError:
                continue
        return None

    @staticmethod
    def _score_to_severity(score: float) -> str:
        """Convert CVSS base score to severity label (CVSS v3 thresholds)."""
        if score == 0.0:
            return "NONE"
        elif score < 4.0:
            return "LOW"
        elif score < 7.0:
            return "MEDIUM"
        elif score < 9.0:
            return "HIGH"
        else:
            return "CRITICAL"


# ─── Feature column lists for ML pipeline ────────────────────────────────────
CVSS_BINARY_FEATURES = [
    "av_network", "av_adjacent", "av_local", "av_physical",
    "ac_low", "ac_high",
    "pr_none", "pr_low", "pr_high",
    "ui_none", "ui_required",
    "scope_changed",
    "ci_none", "ci_low", "ci_high",
    "ii_none", "ii_low", "ii_high",
    "ai_none", "ai_low", "ai_high",
    "has_cvss_v3",
]

TEMPORAL_FEATURES = [
    "days_since_published",
    "quarter_published",
    "year_published",
    "is_recent",
]

EXPLOITATION_FEATURES = [
    "has_exploit_reference",
]

POST_PUB_THREAT_FEATURES = [
    "in_cisa_kev",
    "epss_score",
    "epss_percentile",
]

TEXT_FEATURES = [
    "description_length",
    "description_word_count",
    "has_rce_keyword",
    "has_sqli_keyword",
    "has_xss_keyword",
    "has_overflow_keyword",
    "has_auth_bypass_keyword",
    "has_priv_esc_keyword",
    "has_dos_keyword",
]

CWE_FEATURES = [
    "is_memory_safety_cwe",
    "is_injection_cwe",
    "is_auth_cwe",
    "is_config_cwe",
    "cwe_category_id",
]

METADATA_FEATURES = [
    "vendor_count",
    "reference_count",
    "has_patch_reference",
]

STAGE1_FEATURES = (
    CVSS_BINARY_FEATURES
    + TEMPORAL_FEATURES
    + EXPLOITATION_FEATURES
    + TEXT_FEATURES
    + CWE_FEATURES
    + METADATA_FEATURES
)

ALL_FEATURES = STAGE1_FEATURES

TARGET_COLUMN = "severity_numeric"
TARGET_LABEL_COLUMN = "severity_label"

# Module level alias for compatibility
FeaturePipeline = FeatureEngineer

