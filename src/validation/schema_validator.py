"""
Schema Validator
================
Validates raw and engineered vulnerability data using Pandera schemas.

Catches:
- Missing required fields
- Out-of-range CVSS scores
- Invalid severity labels
- Type mismatches
- Referential integrity issues (e.g., CVE ID format)
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)
CVE_ID_PATTERN = re.compile(r"^CVE-\d{4}-\d{4,}$")

try:
    import pandera as pa

    from pandera import Column, DataFrameSchema, Check
    HAS_PANDERA = True
except ImportError:
    HAS_PANDERA = False
    pa = None
    Column = None
    DataFrameSchema = None
    Check = None

if HAS_PANDERA:
    RAW_VULNERABILITY_SCHEMA = DataFrameSchema(
        columns={
            "cve_id": Column(
                str,
                Check(lambda x: x.str.match(r"^CVE-\d{4}-\d{4,}$").all(),
                      error="CVE ID format invalid"),
                nullable=False,
            ),
            "published_date": Column(
                object,
                nullable=False,
            ),
            "description": Column(
                str,
                nullable=True,
            ),
            "base_severity": Column(
                str,
                Check(
                    lambda x: x.dropna().isin(
                        ["NONE", "LOW", "MEDIUM", "HIGH", "CRITICAL", ""]
                    ).all(),
                    error="Invalid severity label",
                ),
                nullable=True,
            ),
            "base_score": Column(
                float,
                Check(lambda x: (x.dropna() >= 0.0).all() & (x.dropna() <= 10.0).all(),
                      error="CVSS base score out of range [0, 10]"),
                nullable=True,
            ),
        },
        coerce=True,
        strict=False,
    )

    ENGINEERED_FEATURE_SCHEMA = DataFrameSchema(
        columns={
            "cve_id": Column(str, nullable=False),
            "severity_numeric": Column(
                int,
                Check(lambda x: x.dropna().isin([-1, 0, 1, 2, 3, 4]).all(),
                      error="severity_numeric must be -1 to 4"),
                nullable=True,
            ),
            "epss_score": Column(
                float,
                Check(lambda x: (x >= 0.0).all() & (x <= 1.0).all(),
                      error="EPSS score out of range [0, 1]"),
                nullable=False,
            ),
            "epss_percentile": Column(
                float,
                Check(lambda x: (x >= 0.0).all() & (x <= 1.0).all(),
                      error="EPSS percentile out of range [0, 1]"),
                nullable=False,
            ),
            "days_since_published": Column(
                float,
                Check(lambda x: (x.dropna() >= 0).all(),
                      error="days_since_published cannot be negative"),
                nullable=True,
            ),
            "description_length": Column(int, nullable=False),
        },
        coerce=True,
        strict=False,
    )
else:
    RAW_VULNERABILITY_SCHEMA = None
    ENGINEERED_FEATURE_SCHEMA = None



class DataValidator:
    """
    Validates DataFrames at different pipeline stages.

    Produces a structured validation report for monitoring.
    """

    def validate_raw(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """
        Validate raw NVD data.

        Returns:
            (cleaned_df, report_dict) — report contains pass/fail + statistics
        """
        report: dict[str, Any] = {
            "stage": "raw_validation",
            "input_rows": len(df),
            "passed": True,
            "errors": [],
            "warnings": [],
            "statistics": {},
        }

        # Deduplication
        before = len(df)
        df = df.drop_duplicates(subset=["cve_id"], keep="last")
        dupes = before - len(df)
        if dupes > 0:
            report["warnings"].append(f"Removed {dupes} duplicate CVE records")

        # CVE ID format check
        invalid_ids = df[~df["cve_id"].str.match(r"^CVE-\d{4}-\d{4,}$", na=False)]
        if len(invalid_ids) > 0:
            report["warnings"].append(
                f"{len(invalid_ids)} records with non-standard CVE IDs"
            )
            df = df[df["cve_id"].str.match(r"^CVE-\d{4}-\d{4,}$", na=False)]

        # Schema validation
        if HAS_PANDERA and RAW_VULNERABILITY_SCHEMA is not None:
            try:
                RAW_VULNERABILITY_SCHEMA.validate(df, lazy=True)
            except pa.errors.SchemaErrors as e:
                report["errors"].append(str(e.failure_cases.head(10).to_dict()))
                report["passed"] = False


        # Statistics
        report["statistics"] = {
            "total_records": len(df),
            "with_cvss_v3": int(df["cvss_vector_string"].notna().sum()),
            "with_description": int(df["description"].notna().sum()),
            "severity_distribution": (
                df["base_severity"].value_counts().to_dict()
                if "base_severity" in df.columns else {}
            ),
            "year_range": (
                [int(df["published_date"].min()[:4]),
                 int(df["published_date"].max()[:4])]
                if "published_date" in df.columns and len(df) > 0 else [None, None]
            ),
        }

        report["output_rows"] = len(df)
        logger.info(
            f"Raw validation: {len(df)} records, passed={report['passed']}, "
            f"{len(report['errors'])} errors, {len(report['warnings'])} warnings"
        )
        return df, report

    def validate_features(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Validate engineered features DataFrame."""
        report: dict[str, Any] = {
            "stage": "feature_validation",
            "input_rows": len(df),
            "passed": True,
            "errors": [],
            "warnings": [],
            "statistics": {},
        }

        # Check for extreme class imbalance
        if "severity_numeric" in df.columns:
            dist = df["severity_numeric"].value_counts(normalize=True)
            for cls, pct in dist.items():
                if pct < 0.01:
                    report["warnings"].append(
                        f"Class {cls} has only {pct*100:.1f}% of samples"
                    )

        # Check for NaN in critical feature columns
        critical_cols = ["cve_id", "epss_score", "epss_percentile"]
        for col in critical_cols:
            if col in df.columns:
                n_null = df[col].isna().sum()
                if n_null > 0:
                    report["warnings"].append(f"{col}: {n_null} null values")

        # Schema validation
        if HAS_PANDERA and ENGINEERED_FEATURE_SCHEMA is not None:
            try:
                ENGINEERED_FEATURE_SCHEMA.validate(df, lazy=True)
            except pa.errors.SchemaErrors as e:
                report["errors"].append(str(e.failure_cases.head(10).to_dict()))
                report["passed"] = False


        report["statistics"] = {
            "total_features": len(df.columns),
            "total_records": len(df),
            "missing_severity": int((df.get("severity_numeric", pd.Series()) == -1).sum()),
            "in_kev_count": int(df["in_cisa_kev"].sum()) if "in_cisa_kev" in df.columns else 0,
            "high_epss_count": int(
                (df["epss_score"] > 0.1).sum()
            ) if "epss_score" in df.columns else 0,
        }

        report["output_rows"] = len(df)
        logger.info(
            f"Feature validation: {len(df)} records, passed={report['passed']}"
        )
        return df, report

    def check_leakage(self, df: pd.DataFrame, feature_cols: list[str]) -> list[str]:
        """
        Check that no target-leaking columns are present in feature set.

        Returns list of detected leakage columns.
        """
        FORBIDDEN_FEATURES = {
            "base_score", "base_severity",
            "exploitability_score", "impact_score",
            "cvss_v2_score", "cvss_v2_severity",
            "severity_label",  # target label
        }
        leakage = [c for c in feature_cols if c.lower() in FORBIDDEN_FEATURES]
        if leakage:
            logger.error(f"DATA LEAKAGE DETECTED: {leakage}")
        return leakage


class VulnerabilitySchemaValidator(DataValidator):
    """Alias for DataValidator for compatibility across pipeline modules and DAGs."""

    def validate(self, df: pd.DataFrame, bad_row_threshold: float = 0.05) -> pd.DataFrame:
        """Convenience method to validate raw data and raise on threshold error."""
        cleaned_df, report = self.validate_raw(df)
        total = report["input_rows"]
        dropped = total - report["output_rows"]
        if total > 0 and (dropped / total) > bad_row_threshold:
            raise ValueError(
                f"Data validation failed: dropped {dropped}/{total} rows ({dropped/total:.1%}), "
                f"exceeding bad-row threshold of {bad_row_threshold:.1%}. Errors: {report['errors']}"
            )
        return cleaned_df

