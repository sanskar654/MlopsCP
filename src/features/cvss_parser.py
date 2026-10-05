"""
CVSS Vector String Parser
=========================
Parses CVSS v3.0 and v3.1 vector strings into structured submetric dicts.

Example input:  "CVSS:3.1/AV:N/AC:L/PR:H/UI:N/S:U/C:L/I:L/A:N"
Example output: {
    "version": "3.1",
    "attack_vector": "NETWORK",
    "attack_complexity": "LOW",
    "privileges_required": "HIGH",
    "user_interaction": "NONE",
    "scope": "UNCHANGED",
    "confidentiality_impact": "LOW",
    "integrity_impact": "LOW",
    "availability_impact": "NONE",
}

These raw submetrics are our PRIMARY ML features — the base score is NOT used
as a feature because it IS the target (severity label), which would be data leakage.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ─── Metric value mappings ────────────────────────────────────────────────────
ATTACK_VECTOR_MAP = {
    "N": "NETWORK",
    "A": "ADJACENT",
    "L": "LOCAL",
    "P": "PHYSICAL",
}

ATTACK_COMPLEXITY_MAP = {
    "L": "LOW",
    "H": "HIGH",
}

PRIVILEGES_REQUIRED_MAP = {
    "N": "NONE",
    "L": "LOW",
    "H": "HIGH",
}

USER_INTERACTION_MAP = {
    "N": "NONE",
    "R": "REQUIRED",
}

SCOPE_MAP = {
    "U": "UNCHANGED",
    "C": "CHANGED",
}

IMPACT_MAP = {
    "N": "NONE",
    "L": "LOW",
    "H": "HIGH",
}


def parse_cvss_vector(vector_string: str | None) -> dict[str, str | None]:
    """
    Parse a CVSS v3.x vector string into a named-field dict.

    Args:
        vector_string: e.g. "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"

    Returns:
        Dict with all 8 submetric fields expanded to human-readable names.
        Fields are None if parsing fails.
    """
    empty = {
        "version": None,
        "attack_vector": None,
        "attack_complexity": None,
        "privileges_required": None,
        "user_interaction": None,
        "scope": None,
        "confidentiality_impact": None,
        "integrity_impact": None,
        "availability_impact": None,
    }

    if not vector_string:
        return empty

    try:
        parts = vector_string.strip().split("/")
        if len(parts) < 9:
            logger.debug(f"Short CVSS vector: {vector_string}")
            return empty

        # First part is the version prefix: CVSS:3.1
        version_part = parts[0]
        version = version_part.split(":")[1] if ":" in version_part else None

        # Build a lookup from metric abbreviation to value abbreviation
        metrics: dict[str, str] = {}
        for part in parts[1:]:
            if ":" in part:
                key, val = part.split(":", 1)
                metrics[key] = val

        return {
            "version": version,
            "attack_vector": ATTACK_VECTOR_MAP.get(metrics.get("AV", ""), None),
            "attack_complexity": ATTACK_COMPLEXITY_MAP.get(metrics.get("AC", ""), None),
            "privileges_required": PRIVILEGES_REQUIRED_MAP.get(metrics.get("PR", ""), None),
            "user_interaction": USER_INTERACTION_MAP.get(metrics.get("UI", ""), None),
            "scope": SCOPE_MAP.get(metrics.get("S", ""), None),
            "confidentiality_impact": IMPACT_MAP.get(metrics.get("C", ""), None),
            "integrity_impact": IMPACT_MAP.get(metrics.get("I", ""), None),
            "availability_impact": IMPACT_MAP.get(metrics.get("A", ""), None),
        }

    except Exception as e:
        logger.warning(f"Failed to parse CVSS vector '{vector_string}': {e}")
        return empty


def cvss_submetrics_to_onehot(parsed: dict[str, str | None]) -> dict[str, bool]:
    """
    Convert parsed CVSS submetrics into one-hot encoded boolean features.

    These are the features fed to the ML model — no base score, no leakage.

    Args:
        parsed: Output from parse_cvss_vector()

    Returns:
        Dict of boolean one-hot features
    """
    av = parsed.get("attack_vector")
    ac = parsed.get("attack_complexity")
    pr = parsed.get("privileges_required")
    ui = parsed.get("user_interaction")
    s = parsed.get("scope")
    ci = parsed.get("confidentiality_impact")
    ii = parsed.get("integrity_impact")
    ai = parsed.get("availability_impact")

    return {
        # Attack Vector
        "av_network": av == "NETWORK",
        "av_adjacent": av == "ADJACENT",
        "av_local": av == "LOCAL",
        "av_physical": av == "PHYSICAL",
        # Attack Complexity
        "ac_low": ac == "LOW",
        "ac_high": ac == "HIGH",
        # Privileges Required
        "pr_none": pr == "NONE",
        "pr_low": pr == "LOW",
        "pr_high": pr == "HIGH",
        # User Interaction
        "ui_none": ui == "NONE",
        "ui_required": ui == "REQUIRED",
        # Scope
        "scope_changed": s == "CHANGED",
        # Confidentiality Impact
        "ci_none": ci == "NONE",
        "ci_low": ci == "LOW",
        "ci_high": ci == "HIGH",
        # Integrity Impact
        "ii_none": ii == "NONE",
        "ii_low": ii == "LOW",
        "ii_high": ii == "HIGH",
        # Availability Impact
        "ai_none": ai == "NONE",
        "ai_low": ai == "LOW",
        "ai_high": ai == "HIGH",
        # Has CVSS at all
        "has_cvss_v3": av is not None,
    }


def severity_to_numeric(severity: str | None) -> int:
    """Convert severity label to ordinal integer (0–4)."""
    mapping = {
        "NONE": 0,
        "LOW": 1,
        "MEDIUM": 2,
        "HIGH": 3,
        "CRITICAL": 4,
    }
    return mapping.get((severity or "").upper(), -1)


NUMERIC_TO_SEVERITY = {0: "NONE", 1: "LOW", 2: "MEDIUM", 3: "HIGH", 4: "CRITICAL"}
