"""
Feature Engineering Pipeline (Leakage-Free & Symptom-Based)
===========================================================
Transforms raw vulnerability records into ML-ready features before publication.

Core Principles:
1. NO CVSS vector submetrics (AV, AC, PR, UI, S, C, I, A). CVSS score is derived
   directly from those, so using them amounts to copying the formula.
2. NO post-publication leakage (EPSS score, CISA KEV listing, CVSS base score).
3. Rich features built strictly from symptoms available at discovery:
   - TF-IDF n-gram vectorization on description text
   - Comprehensive security attack keyword flags (RCE, SQLi, Buffer Overflow, etc.)
   - Description length and word count
   - CWE category grouping & classification
   - Vendor / Product recognition
   - Reference count and reference URL patterns
   - Publication calendar time (year, quarter)
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

FEATURE_VERSION = "2.0"
TFIDF_MAX_FEATURES = 100
VECTORIZER_PATH = Path("configs/tfidf_vectorizer.joblib")

# ─── Attack Keyword Patterns ─────────────────────────────────────────────────
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
        r"\b(buffer.?overflow|stack.?overflow|heap.?overflow|integer.?overflow|memory.?corrupt|use.?after.?free|double.?free)\b",
        re.IGNORECASE,
    ),
    "has_auth_bypass_keyword": re.compile(
        r"\b(auth.?bypass|authentication.?bypass|bypass.?auth|unauthenticated|improper.?auth|session.?fixation)\b",
        re.IGNORECASE,
    ),
    "has_priv_esc_keyword": re.compile(
        r"\b(privilege.?escal|priv.?esc|local.?priv|gain.?privilege|elevat.?privilege|root.?access)\b",
        re.IGNORECASE,
    ),
    "has_dos_keyword": re.compile(
        r"\b(denial.?of.?service|dos|infinite.?loop|crash|hung|hang|resource.?exhaust)\b",
        re.IGNORECASE,
    ),
    "has_traversal_keyword": re.compile(
        r"\b(path.?traversal|directory.?traversal|arbitrary.?file|file.?read)\b",
        re.IGNORECASE,
    ),
    "has_ssrf_keyword": re.compile(
        r"\b(server.?side.?request.?forgery|ssrf)\b",
        re.IGNORECASE,
    ),
    "has_info_leak_keyword": re.compile(
        r"\b(information.?disclosure|information.?leak|sensitive.?data|leak.?memory)\b",
        re.IGNORECASE,
    ),
}

# ─── CWE Category Groupings ──────────────────────────────────────────────────
MEMORY_SAFETY_CWES = {
    "CWE-119",
    "CWE-120",
    "CWE-121",
    "CWE-122",
    "CWE-125",
    "CWE-126",
    "CWE-127",
    "CWE-415",
    "CWE-416",
    "CWE-787",
    "CWE-788",
    "CWE-824",
    "CWE-476",
}

INJECTION_CWES = {
    "CWE-77",
    "CWE-78",
    "CWE-79",
    "CWE-89",
    "CWE-90",
    "CWE-91",
    "CWE-94",
    "CWE-95",
    "CWE-96",
    "CWE-113",
    "CWE-917",
    "CWE-943",
}

AUTH_CWES = {
    "CWE-287",
    "CWE-288",
    "CWE-290",
    "CWE-294",
    "CWE-295",
    "CWE-306",
    "CWE-307",
    "CWE-384",
    "CWE-522",
    "CWE-798",
}

CONFIG_CWES = {
    "CWE-16",
    "CWE-183",
    "CWE-185",
    "CWE-200",
    "CWE-209",
    "CWE-250",
    "CWE-276",
    "CWE-732",
    "CWE-749",
}

CRYPTO_CWES = {
    "CWE-310",
    "CWE-311",
    "CWE-326",
    "CWE-327",
    "CWE-330",
    "CWE-338",
}

PRIVILEGE_CWES = {
    "CWE-264",
    "CWE-269",
    "CWE-272",
    "CWE-274",
    "CWE-732",
}

# ─── Major Enterprise Vendors ────────────────────────────────────────────────
TOP_VENDORS = [
    "microsoft",
    "apple",
    "google",
    "cisco",
    "linux",
    "oracle",
    "apache",
    "adobe",
    "ibm",
    "vmware",
    "redhat",
    "mozilla",
]

# ─── Pre-computed Non-Text Features ──────────────────────────────────────────
TEXT_METRIC_FEATURES = [
    "description_length",
    "description_word_count",
    "has_rce_keyword",
    "has_sqli_keyword",
    "has_xss_keyword",
    "has_overflow_keyword",
    "has_auth_bypass_keyword",
    "has_priv_esc_keyword",
    "has_dos_keyword",
    "has_traversal_keyword",
    "has_ssrf_keyword",
    "has_info_leak_keyword",
]

CWE_FEATURES = [
    "is_memory_safety_cwe",
    "is_injection_cwe",
    "is_auth_cwe",
    "is_config_cwe",
    "is_crypto_cwe",
    "is_privilege_cwe",
    "cwe_category_id",
]

VENDOR_FEATURES = [f"vendor_{v}" for v in TOP_VENDORS] + [
    "vendor_other",
    "vendor_count",
]

REFERENCE_FEATURES = [
    "reference_count",
    "has_patch_reference",
    "has_exploit_reference",
]

TEMPORAL_FEATURES = [
    "year_published",
    "quarter_published",
    "days_since_published",
    "is_recent",
]

TABULAR_FEATURES = (
    TEXT_METRIC_FEATURES
    + CWE_FEATURES
    + VENDOR_FEATURES
    + REFERENCE_FEATURES
    + TEMPORAL_FEATURES
)

# TFIDF features named tfidf_0 .. tfidf_99
TFIDF_FEATURE_NAMES = [f"tfidf_{i}" for i in range(TFIDF_MAX_FEATURES)]

# ALL_FEATURES strictly excludes any CVSS submetrics or post-pub signals
ALL_FEATURES = TABULAR_FEATURES + TFIDF_FEATURE_NAMES

SEVERITY_MAP = {
    "LOW": 0,
    "MEDIUM": 1,
    "HIGH": 2,
    "CRITICAL": 3,
}

REVERSE_SEVERITY_MAP = {v: k for k, v in SEVERITY_MAP.items()}


class FeatureEngineer:
    """
    Leakage-free Feature Engineering Pipeline.
    Fits TF-IDF vectorizer on training text and transforms vulnerability records
    into clean, symptom-based numeric matrices.
    """

    def __init__(self, reference_date: Optional[datetime] = None):
        self.reference_date = reference_date or datetime.now(timezone.utc)
        self.vectorizer: Optional[TfidfVectorizer] = None
        self._load_vectorizer_if_available()

    def _load_vectorizer_if_available(self) -> None:
        if VECTORIZER_PATH.exists():
            try:
                self.vectorizer = joblib.load(VECTORIZER_PATH)
                logger.info("Loaded fitted TF-IDF vectorizer from %s", VECTORIZER_PATH)
            except Exception as e:
                logger.debug("Could not load vectorizer: %s", e)

    def fit(self, df_raw: pd.DataFrame) -> "FeatureEngineer":
        """Fit the TF-IDF vectorizer on descriptions."""
        texts = (
            df_raw.get("description", pd.Series(dtype=str))
            .fillna("")
            .astype(str)
            .tolist()
        )
        self.vectorizer = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=(1, 2),
            stop_words="english",
            lowercase=True,
        )
        self.vectorizer.fit(texts)
        VECTORIZER_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, VECTORIZER_PATH)
        logger.info(
            "Fitted and saved TF-IDF vectorizer with %d features", TFIDF_MAX_FEATURES
        )
        return self

    def transform(
        self,
        df_raw: pd.DataFrame,
        kev_cve_ids: Optional[Set[str]] = None,
        epss_scores: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> pd.DataFrame:
        """
        Transform raw vulnerability records into feature matrix.
        Post-publication parameters (kev_cve_ids, epss_scores) are accepted for backwards compatibility
        and stored in separate metadata columns, NEVER in feature matrix X.
        """
        if self.vectorizer is None:
            self.fit(df_raw)

        records: List[Dict[str, Any]] = []
        descriptions: List[str] = []

        for _, row in df_raw.iterrows():
            rec = self._engineer_tabular_features(row.to_dict())
            if rec is not None:
                records.append(rec)
                descriptions.append(str(row.get("description") or ""))

        df_tabular = pd.DataFrame(records)
        if df_tabular.empty:
            return pd.DataFrame(columns=["cve_id"] + ALL_FEATURES)

        # Extract TF-IDF features
        if self.vectorizer is not None and descriptions:
            tfidf_mat = self.vectorizer.transform(descriptions).toarray()
            # If fitted vocabulary is smaller than max features, pad
            if tfidf_mat.shape[1] < TFIDF_MAX_FEATURES:
                padding = np.zeros(
                    (tfidf_mat.shape[0], TFIDF_MAX_FEATURES - tfidf_mat.shape[1])
                )
                tfidf_mat = np.hstack([tfidf_mat, padding])
            tfidf_df = pd.DataFrame(
                tfidf_mat[:, :TFIDF_MAX_FEATURES], columns=TFIDF_FEATURE_NAMES
            )
        else:
            tfidf_df = pd.DataFrame(
                0.0, index=df_tabular.index, columns=TFIDF_FEATURE_NAMES
            )

        df_features = pd.concat([df_tabular, tfidf_df], axis=1)

        # Ensure all columns present
        for col in ALL_FEATURES:
            if col not in df_features.columns:
                df_features[col] = 0

        logger.info(
            "Engineered %d features for %d records (Leakage-free v2.0)",
            len(ALL_FEATURES),
            len(df_features),
        )
        return df_features

    def fit_transform(self, df_raw: pd.DataFrame, y=None) -> pd.DataFrame:
        """Fit vectorizer on data and return feature matrix X (scikit-learn compatible)."""
        self.fit(df_raw)
        df = self.transform(df_raw)
        # Drop metadata and target columns to return pure X
        cols_to_drop = [
            c
            for c in [
                "cve_id",
                "severity_label",
                "severity_numeric",
                "base_severity",
                "base_score",
                "cvss_score",
                "cvss_severity",
                "epss_score",
                "epss_percentile",
                "in_cisa_kev",
                "feature_version",
            ]
            if c in df.columns
        ]
        return df.drop(columns=cols_to_drop)

    def engineer_features(self, df_raw: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Alias for transform."""
        return self.transform(df_raw)

    def _engineer_tabular_features(
        self, row: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        cve_id = row.get("cve_id") or ""
        if not cve_id:
            return None

        features: Dict[str, Any] = {
            "cve_id": cve_id,
            "feature_version": FEATURE_VERSION,
        }

        # ─── Text / NLP Metrics ──────────────────────────────────────────────
        desc = str(row.get("description") or "")
        features["description_length"] = len(desc)
        features["description_word_count"] = len(desc.split())

        for feat_name, pattern in KEYWORD_PATTERNS.items():
            features[feat_name] = int(bool(pattern.search(desc)))

        # ─── CWE Features ────────────────────────────────────────────────────
        cwe_raw = row.get("cwe_ids") or row.get("cwe") or []
        if isinstance(cwe_raw, str):
            cwe_list = re.findall(r"CWE-\d+", cwe_raw.upper())
        elif isinstance(cwe_raw, (list, set)):
            cwe_list = [str(c).upper() for c in cwe_raw if "CWE" in str(c).upper()]
        else:
            cwe_list = []

        cwe_set = set(cwe_list)
        features["is_memory_safety_cwe"] = int(bool(cwe_set & MEMORY_SAFETY_CWES))
        features["is_injection_cwe"] = int(bool(cwe_set & INJECTION_CWES))
        features["is_auth_cwe"] = int(bool(cwe_set & AUTH_CWES))
        features["is_config_cwe"] = int(bool(cwe_set & CONFIG_CWES))
        features["is_crypto_cwe"] = int(bool(cwe_set & CRYPTO_CWES))
        features["is_privilege_cwe"] = int(bool(cwe_set & PRIVILEGE_CWES))

        # First CWE ID as numeric integer
        cwe_num = 0
        for cwe in cwe_set:
            digits = re.findall(r"\d+", cwe)
            if digits:
                cwe_num = int(digits[0])
                break
        features["cwe_category_id"] = cwe_num

        # ─── Vendor / Product Recognition ────────────────────────────────────
        vendor_text = (
            str(row.get("vendor") or "")
            + " "
            + str(row.get("product") or "")
            + " "
            + str(row.get("cpe_list") or "")
            + " "
            + desc
        ).lower()

        matched_any_vendor = False
        for vendor in TOP_VENDORS:
            is_match = int(vendor in vendor_text)
            features[f"vendor_{vendor}"] = is_match
            if is_match:
                matched_any_vendor = True

        features["vendor_other"] = 0 if matched_any_vendor else 1
        features["vendor_count"] = int(
            row.get("vendor_count") or (1 if matched_any_vendor else 0)
        )

        # ─── Reference Features ──────────────────────────────────────────────
        ref_urls = row.get("reference_urls") or []
        if isinstance(ref_urls, str):
            ref_urls = [ref_urls]
        ref_text = " ".join([str(u).lower() for u in ref_urls])

        features["reference_count"] = int(row.get("reference_count") or len(ref_urls))
        features["has_patch_reference"] = int(
            any(
                k in ref_text
                for k in ["patch", "advisory", "commit", "update", "fix", "security"]
            )
        )
        features["has_exploit_reference"] = int(
            any(
                k in ref_text
                for k in ["exploit", "poc", "metasploit", "exploit-db", "packetstorm"]
            )
        )

        # ─── Temporal Features ───────────────────────────────────────────────
        published = self._parse_dt(row.get("published_date"))
        if published:
            delta = (
                self.reference_date - published.replace(tzinfo=timezone.utc)
                if published.tzinfo is None
                else self.reference_date - published
            )
            features["days_since_published"] = max(0, delta.days)
            features["quarter_published"] = (published.month - 1) // 3 + 1
            features["year_published"] = published.year
            features["is_recent"] = int(delta.days < 90)
        else:
            features["days_since_published"] = 0
            features["quarter_published"] = 1
            features["year_published"] = 2024
            features["is_recent"] = 0

        # ─── Target Severity Label ───────────────────────────────────────────
        raw_sev = (
            str(row.get("base_severity") or row.get("cvss_severity") or "")
            .upper()
            .strip()
        )
        if not raw_sev and row.get("base_score") is not None:
            try:
                score = float(row["base_score"])
                if score >= 9.0:
                    raw_sev = "CRITICAL"
                elif score >= 7.0:
                    raw_sev = "HIGH"
                elif score >= 4.0:
                    raw_sev = "MEDIUM"
                else:
                    raw_sev = "LOW"
            except (ValueError, TypeError):
                pass

        if raw_sev in SEVERITY_MAP:
            features["severity_label"] = raw_sev
            features["severity_numeric"] = SEVERITY_MAP[raw_sev]
        else:
            features["severity_label"] = None
            features["severity_numeric"] = -1

        return features

    @staticmethod
    def _parse_dt(dt_val: Any) -> Optional[datetime]:
        if not dt_val:
            return None
        if isinstance(dt_val, datetime):
            return dt_val
        try:
            val_str = str(dt_val).split(".")[0].replace("Z", "")
            return datetime.fromisoformat(val_str)
        except Exception:
            return None


TARGET_COLUMN = "severity_numeric"
TARGET_LABEL_COLUMN = "severity_label"
FeaturePipeline = FeatureEngineer
