"""
Pydantic Schemas
================
Request and response models for all API endpoints.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


# ─── Prediction ──────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    cve_id: str = Field(..., description="CVE identifier", example="CVE-2021-44228")
    cvss_vector_string: Optional[str] = Field(
        None,
        description="CVSS v3.x vector string",
        example="CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H",
    )
    description: Optional[str] = Field(
        None, description="CVE description text for NLP features"
    )
    epss_score: float = Field(0.0, ge=0.0, le=1.0, description="EPSS exploitation probability")
    epss_percentile: float = Field(0.0, ge=0.0, le=1.0)
    in_cisa_kev: bool = Field(False, description="Is this CVE in CISA KEV catalog")
    year_published: Optional[int] = Field(None, ge=1999, le=2030)
    cwe_ids: list[str] = Field(default_factory=list)
    reference_count: int = Field(0, ge=0)
    vendor_count: int = Field(0, ge=0)


class PredictResponse(BaseModel):
    cve_id: str
    predicted_severity: str
    predicted_class: int
    confidence: float
    probabilities: dict[str, float]
    model_version: str
    prediction_id: str


# ─── Risk Scoring ─────────────────────────────────────────────────────────────

class RiskScoreRequest(BaseModel):
    cve_id: str
    cvss_vector_string: Optional[str] = None
    description: Optional[str] = None
    epss_score: float = Field(0.0, ge=0.0, le=1.0)
    epss_percentile: float = Field(0.0, ge=0.0, le=1.0)
    in_cisa_kev: bool = False
    year_published: Optional[int] = None
    cwe_ids: list[str] = Field(default_factory=list)
    reference_count: int = 0
    vendor_count: int = 0
    asset_criticality: float = Field(
        0.5, ge=0.0, le=1.0,
        description="Organizational asset importance (0=low, 1=critical)"
    )
    exposure_score: float = Field(
        0.5, ge=0.0, le=1.0,
        description="Network exposure (0=air-gapped, 1=internet-facing)"
    )


class RiskComponentsResponse(BaseModel):
    cvss_component: float
    kev_component: float
    epss_component: float
    asset_criticality: float
    exposure_score: float
    confidence_component: float


class RiskScoreResponse(BaseModel):
    cve_id: str
    risk_score: float = Field(..., description="Composite risk score 0–100")
    risk_tier: str = Field(..., description="LOW_RISK|MEDIUM_RISK|HIGH_RISK|CRITICAL_RISK")
    predicted_severity: str
    model_confidence: float
    components: RiskComponentsResponse
    component_contributions: dict[str, float]
    weights_used: dict[str, float]
    model_version: str


# ─── Explanation ──────────────────────────────────────────────────────────────

class ExplainRequest(BaseModel):
    cve_id: str
    cvss_vector_string: Optional[str] = None
    description: Optional[str] = None
    epss_score: float = 0.0
    epss_percentile: float = 0.0
    in_cisa_kev: bool = False
    year_published: Optional[int] = None
    cwe_ids: list[str] = Field(default_factory=list)
    reference_count: int = 0
    vendor_count: int = 0
    top_n_features: int = Field(10, ge=1, le=30)


class FeatureContribution(BaseModel):
    feature: str
    shap_value: Optional[float]
    feature_value: Optional[float]
    direction: Optional[str]


class ExplainResponse(BaseModel):
    cve_id: str
    predicted_severity: str
    predicted_class: int
    confidence: float
    base_value: Optional[float]
    top_features: list[FeatureContribution]
    explanation_text: str
    total_shap_sum: Optional[float]


# ─── Vulnerability List ───────────────────────────────────────────────────────

class VulnerabilityListItem(BaseModel):
    cve_id: str
    description: Optional[str]
    published_date: Optional[str]
    base_severity: Optional[str]
    base_score: Optional[float]
    predicted_severity: Optional[str]
    risk_score: Optional[float]
    risk_tier: Optional[str]
    in_cisa_kev: bool = False
    epss_score: Optional[float]
    model_confidence: Optional[float]


class VulnerabilityListResponse(BaseModel):
    items: list[VulnerabilityListItem]
    total: int
    page: int
    page_size: int
    pages: int


class VulnerabilityDetailResponse(BaseModel):
    cve_id: str
    description: Optional[str]
    published_date: Optional[str]
    last_modified_date: Optional[str]
    vuln_status: Optional[str]
    base_severity: Optional[str]
    base_score: Optional[float]
    cvss_vector_string: Optional[str]
    attack_vector: Optional[str]
    attack_complexity: Optional[str]
    privileges_required: Optional[str]
    user_interaction: Optional[str]
    scope: Optional[str]
    confidentiality_impact: Optional[str]
    integrity_impact: Optional[str]
    availability_impact: Optional[str]
    cwe_ids: list[str] = []
    reference_urls: list[str] = []
    in_cisa_kev: bool = False
    epss_score: Optional[float]
    epss_percentile: Optional[float]
    predicted_severity: Optional[str]
    risk_score: Optional[float]
    risk_tier: Optional[str]
    model_confidence: Optional[float]


# ─── Feedback ────────────────────────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    cve_id: str
    analyst_id: str = Field(..., description="Analyst identifier")
    predicted_severity: Optional[str] = None
    corrected_severity: str = Field(
        ...,
        description="Analyst's corrected severity label",
        pattern="^(NONE|LOW|MEDIUM|HIGH|CRITICAL)$",
    )
    feedback_notes: Optional[str] = None
    confidence_rating: int = Field(
        3, ge=1, le=5,
        description="Analyst confidence in correction (1=low, 5=high)"
    )


class FeedbackResponse(BaseModel):
    feedback_id: int
    cve_id: str
    status: str
    message: str


# ─── Model Registry ───────────────────────────────────────────────────────────

class ModelInfo(BaseModel):
    model_name: str
    model_version: str
    model_role: str
    macro_f1: Optional[float]
    critical_recall: Optional[float]
    ece: Optional[float]
    roc_auc: Optional[float]
    latency_p99_ms: Optional[float]
    training_date: Optional[str]
    promoted_at: Optional[str]
    mlflow_run_id: Optional[str]


class ModelCompareResponse(BaseModel):
    champion: Optional[ModelInfo]
    challenger: Optional[ModelInfo]
    comparison: Optional[dict[str, Any]]
    recommendation: Optional[str]


# ─── Drift ────────────────────────────────────────────────────────────────────

class DriftReportResponse(BaseModel):
    report_date: str
    report_type: str
    dataset_drift: Optional[bool]
    drift_score: Optional[float]
    n_drifted_features: int
    drifted_features: list[str]
    trigger_retrain: bool
    report_path: Optional[str]


# ─── Metrics ──────────────────────────────────────────────────────────────────

class ModelMetricsResponse(BaseModel):
    champion_metrics: Optional[dict[str, Any]]
    recent_runs: list[dict[str, Any]]
    drift_summary: Optional[dict[str, Any]]
    feedback_summary: Optional[dict[str, Any]]


# ─── Operations ───────────────────────────────────────────────────────────────

class RetrainRequest(BaseModel):
    reason: str = Field(..., description="Reason for triggering retraining")
    force: bool = Field(False, description="Force retrain even if drift not detected")
    admin_token: str = Field(..., description="Admin authorization token")


class RetrainResponse(BaseModel):
    status: str
    message: str
    airflow_dag_run_id: Optional[str]
    triggered_at: str


# ─── Health ───────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool
    champion_model: Optional[str]
    challenger_model: Optional[str]
    database: str
    mlflow: str
    timestamp: str
