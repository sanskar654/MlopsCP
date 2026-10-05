"""
Risk Scorer
===========
Computes composite risk prioritization scores combining:
- ML-predicted severity (CVSS submetric based)
- CISA KEV (known exploitation confirmed)
- EPSS (exploitation probability)
- Asset criticality (organizational context)
- Network exposure
- Model confidence

Formula:
    risk_score (0–100) = weighted_sum(normalized_components) × 100

All components normalized to [0, 1] before weighting.
Weights configurable via configs/risk_weights.yaml.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default weights (sum = 1.0). Can be overridden via config.
DEFAULT_WEIGHTS = {
    "cvss": 0.30,
    "kev": 0.25,
    "epss": 0.20,
    "asset_criticality": 0.15,
    "exposure": 0.07,
    "model_confidence": 0.03,
}

# Risk tier thresholds (on 0–100 scale)
RISK_TIERS = [
    (90, "CRITICAL_RISK"),
    (70, "HIGH_RISK"),
    (40, "MEDIUM_RISK"),
    (0,  "LOW_RISK"),
]

# CVSS severity to normalized score map
SEVERITY_TO_NORM = {
    0: 0.00,  # NONE
    1: 0.20,  # LOW
    2: 0.50,  # MEDIUM
    3: 0.78,  # HIGH
    4: 1.00,  # CRITICAL
}


@dataclass
class RiskComponents:
    """All normalized components (0–1 each) for a single CVE."""
    cve_id: str
    cvss_component: float = 0.0      # Severity score normalized
    kev_component: float = 0.0       # 1.0 if in KEV, else 0.0
    epss_component: float = 0.0      # Raw EPSS score (already 0–1)
    asset_criticality: float = 0.5   # Org asset importance (0=unimportant, 1=critical)
    exposure_score: float = 0.5      # Network exposure (0=air-gapped, 1=internet-facing)
    confidence_component: float = 0.5  # Model confidence
    predicted_severity: int = -1
    predicted_severity_label: str = "UNKNOWN"


@dataclass
class RiskResult:
    """Full risk scoring result for a CVE."""
    cve_id: str
    risk_score: float           # 0–100
    risk_tier: str              # LOW_RISK|MEDIUM_RISK|HIGH_RISK|CRITICAL_RISK
    components: RiskComponents
    weights_used: dict[str, float]
    component_contributions: dict[str, float]  # Weighted contribution per component


class RiskScorer:
    """
    Composite risk scorer for vulnerability prioritization.

    Usage:
        scorer = RiskScorer(weights=custom_weights)
        result = scorer.score(
            cve_id="CVE-2021-44228",
            predicted_severity=4,
            model_confidence=0.97,
            in_cisa_kev=True,
            epss_score=0.975,
            asset_criticality=0.9,
            exposure_score=1.0,
        )
        print(result.risk_score)  # e.g., 96.8
    """

    def __init__(self, weights: dict[str, float] | None = None):
        self.weights = weights or DEFAULT_WEIGHTS.copy()
        self._validate_weights()

    def _validate_weights(self) -> None:
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.001:
            logger.warning(
                f"Risk weights sum to {total:.3f}, not 1.0. Normalizing."
            )
            for k in self.weights:
                self.weights[k] /= total

    def score(
        self,
        cve_id: str,
        predicted_severity: int,
        model_confidence: float,
        in_cisa_kev: bool,
        epss_score: float,
        asset_criticality: float = 0.5,
        exposure_score: float = 0.5,
    ) -> RiskResult:
        """
        Compute risk score for a single vulnerability.

        Args:
            cve_id: CVE identifier
            predicted_severity: Predicted severity class (0–4)
            model_confidence: Model's confidence in prediction (0–1)
            in_cisa_kev: Is this CVE in CISA KEV catalog
            epss_score: EPSS exploitation probability (0–1)
            asset_criticality: Org asset criticality (0–1, default 0.5)
            exposure_score: Network exposure (0=air-gapped, 1=internet, default 0.5)

        Returns:
            RiskResult with score, tier, and full component breakdown
        """
        # Build components
        components = RiskComponents(
            cve_id=cve_id,
            cvss_component=SEVERITY_TO_NORM.get(predicted_severity, 0.5),
            kev_component=1.0 if in_cisa_kev else 0.0,
            epss_component=np.clip(epss_score, 0.0, 1.0),
            asset_criticality=np.clip(asset_criticality, 0.0, 1.0),
            exposure_score=np.clip(exposure_score, 0.0, 1.0),
            confidence_component=np.clip(model_confidence, 0.0, 1.0),
            predicted_severity=predicted_severity,
        )

        # Weighted sum
        component_values = {
            "cvss": components.cvss_component,
            "kev": components.kev_component,
            "epss": components.epss_component,
            "asset_criticality": components.asset_criticality,
            "exposure": components.exposure_score,
            "model_confidence": components.confidence_component,
        }

        contributions = {
            k: self.weights[k] * component_values[k]
            for k in self.weights
        }

        raw_score = sum(contributions.values())  # 0–1
        risk_score = round(raw_score * 100, 2)   # 0–100

        # Determine tier
        risk_tier = "LOW_RISK"
        for threshold, tier in RISK_TIERS:
            if risk_score >= threshold:
                risk_tier = tier
                break

        # Scale contributions to percentage of total score for visualization
        component_contributions = {
            k: round(v * 100, 2) for k, v in contributions.items()
        }

        return RiskResult(
            cve_id=cve_id,
            risk_score=risk_score,
            risk_tier=risk_tier,
            components=components,
            weights_used=self.weights.copy(),
            component_contributions=component_contributions,
        )

    def calculate_risk(
        self,
        cve_id: str = "CVE-2021-44228",
        predicted_severity: str | int = "CRITICAL",
        exploitability_subscore: float = 8.0,
        cisa_kev_active: bool = True,
        epss_score: float = 0.95,
        asset_criticality: float = 0.9,
        asset_exposure: float = 1.0,
    ) -> RiskResult:
        """Alias for score() accepting severity label string or integer."""
        sev_map = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
        if isinstance(predicted_severity, str):
            sev_num = sev_map.get(predicted_severity.upper(), 3)
        else:
            sev_num = int(predicted_severity)

        return self.score(
            cve_id=cve_id,
            predicted_severity=sev_num,
            model_confidence=0.95,
            in_cisa_kev=cisa_kev_active,
            epss_score=epss_score,
            asset_criticality=min(1.0, asset_criticality / 5.0) if asset_criticality > 1.0 else asset_criticality,
            exposure_score=min(1.0, asset_exposure / 5.0) if asset_exposure > 1.0 else asset_exposure,
        )


    def score_batch(
        self,
        df: pd.DataFrame,
        severity_predictions: list[dict[str, Any]],
        epss_map: dict[str, float],
        kev_set: set[str],
        asset_map: dict[str, float] | None = None,
        exposure_map: dict[str, float] | None = None,
    ) -> list[RiskResult]:
        """
        Score a batch of CVEs.

        Args:
            df: DataFrame with cve_id column
            severity_predictions: List of prediction dicts from SeverityClassifier
            epss_map: {cve_id: epss_score}
            kev_set: Set of CVE IDs in CISA KEV
            asset_map: {cve_id: asset_criticality} (defaults to 0.5)
            exposure_map: {cve_id: exposure_score} (defaults to 0.5)
        """
        results = []
        asset_map = asset_map or {}
        exposure_map = exposure_map or {}

        for i, (_, row) in enumerate(df.iterrows()):
            cve_id = row["cve_id"]
            pred = severity_predictions[i] if i < len(severity_predictions) else {}

            result = self.score(
                cve_id=cve_id,
                predicted_severity=pred.get("predicted_class", 2),
                model_confidence=pred.get("confidence", 0.5),
                in_cisa_kev=cve_id in kev_set,
                epss_score=epss_map.get(cve_id, 0.0),
                asset_criticality=asset_map.get(cve_id, 0.5),
                exposure_score=exposure_map.get(cve_id, 0.5),
            )
            results.append(result)

        return results

    def to_dataframe(self, results: list[RiskResult]) -> pd.DataFrame:
        """Convert list of RiskResults to a DataFrame for storage."""
        rows = []
        for r in results:
            row = {
                "cve_id": r.cve_id,
                "risk_score": r.risk_score,
                "risk_tier": r.risk_tier,
                "predicted_severity": r.components.predicted_severity,
                "model_confidence": r.components.confidence_component,
                "cvss_component": r.components.cvss_component,
                "kev_component": r.components.kev_component,
                "epss_component": r.components.epss_component,
                "asset_criticality": r.components.asset_criticality,
                "exposure_score": r.components.exposure_score,
            }
            row.update({
                f"contribution_{k}": v
                for k, v in r.component_contributions.items()
            })
            rows.append(row)
        return pd.DataFrame(rows)
