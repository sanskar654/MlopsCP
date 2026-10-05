# SHLOKv2 — System Architecture, ML Lifecycle, Progress & Team Directives

> **Project Title:** Adaptive MLOps for Software Vulnerability Severity Classification & Risk Prioritization  
> **Workspace Root:** `MlopsCP-main`  
> **Author / Maintainer:** Shlok (Data Engineering & Infrastructure Lead)  
> **Target Audience:** Sanskar (ML & MLOps Lead), Chinmayi (API & Frontend Lead), System Evaluators  
> **Last Updated:** September 2026  

---

## 1. Executive Summary & Audit Background (What `sanskarv1.txt` Discovered)

In the initial technical audit documented in `sanskarv1.txt`, a comprehensive inspection of the legacy codebase was performed. While the legacy project presented an ambitious architecture (FastAPI, Airflow, MLflow, Chart.js dashboard), deep technical code inspection revealed critical design flaws, target leakage, broken pipeline DAGs, and mock responses.

### Key Audit Findings from `sanskarv1.txt`:
1. **Target & Temporal Data Leakage:**  
   The legacy feature engineering pipeline included post-publication threat intelligence signals (`epss_score`, `epss_percentile`, `kev_flag`) and CVSS base scores directly into the Stage 1 machine learning training matrix. Because EPSS scores and CISA KEV listings are populated days or months *after* a vulnerability is published, using them to predict initial severity constituted severe temporal leakage.
2. **Synthetic Data Flaws & Ineffective Temporal Validation:**  
   The synthetic dataset generator (`scripts/seed_data.py`) generated dataset records by repeating 8 static vulnerability templates with randomized timestamps. As a result, the same vulnerability templates appeared in both the training set and test set during temporal split cross-validation, invalidating evaluation metrics.
3. **Mock REST API Endpoints:**  
   Multiple FastAPI endpoints (`/models`, `/drift`, `/vulnerabilities`, `/retrain`) returned static, hardcoded JSON objects rather than querying PostgreSQL or MLflow tracking artifacts. The system looked functional on the surface, but the serving layer was detached from database state.
4. **Broken Orchestration DAGs:**  
   Airflow DAGs (`data_ingestion_dag.py`, `model_training_dag.py`, `drift_monitoring_dag.py`) suffered from severe module import errors and function signature mismatches (e.g. calling `fetch_recent_cves()` instead of `fetch_cves_since()`, importing non-existent `VulnerabilitySchemaValidator` and `features.pipeline` modules).
5. **Scope & Title Misalignment:**  
   The legacy documentation claimed "Software Vulnerability Detection" (suggesting static SAST analysis or code scanning). In reality, the project ingests already known CVEs from NVD/CISA/EPSS to predict **CVSS Base Severity Ratings** and compute **Composite Risk Scores**. The title and scope have been clarified accordingly.

---

## 2. Project Vision & Full Pipeline Architecture (What the Project Does Now)

The refactored platform is an end-to-end, enterprise-grade MLOps system that ingests published CVE vulnerabilities, applies leakage-free XGBoost severity classification, calculates dynamic multi-factor risk scores, serves inferences via FastAPI, visualizes results on an executive dashboard, and monitors dataset drift to trigger automated model retraining.

```
                    +-------------------------------------------------------+
                    |                 DATA INGESTION LAYER                  |
                    | NVD 2.0 API  |  FIRST EPSS Bulk CSV  |  CISA KEV JSON |
                    +---------------------------+---------------------------+
                                                |
                                                v
                    +-------------------------------------------------------+
                    |             POSTGRESQL DATA ARCHITECTURE              |
                    | cves | epss_history (Append Only) | kev_history (Append) |
                    | asset_inventory | asset_cve_map | predictions_log     |
                    +---------------------------+---------------------------+
                                                |
                                                v
                    +-------------------------------------------------------+
                    |               DATA VALIDATION & FEATURE               |
                    | Pandera Schema Validation | CVSS Vector Parsing (22)  |
                    | Strict Stage 1 Leakage Guard (No EPSS/KEV in Train)   |
                    +---------------------------+---------------------------+
                                                |
                                                v
                    +-------------------------------------------------------+
                    |           STAGE 1 ML SEVERITY CLASSIFIER              |
                    | Temporal Train/Test Split | SMOTE Class Balancing     |
                    | Calibrated XGBoost | MLflow Experiment Tracking       |
                    +---------------------------+---------------------------+
                                                |
                                                v
                    +-------------------------------------------------------+
                    |            STAGE 2 COMPOSITE RISK SCORER              |
                    | Risk Score (0-100) = 0.30*ML + 0.25*EPSS + 0.25*KEV    |
                    |                    + 0.15*Asset + 0.05*Exposure       |
                    +---------------------------+---------------------------+
                                                |
                                                v
                    +-------------------------------------------------------+
                    |                SERVING & DASHBOARD                    |
                    | FastAPI REST API | Real DB/MLflow Backed Endpoints   |
                    | 5-Page Executive Web Dashboard | SHAP Waterfall Panel |
                    +---------------------------+---------------------------+
                                                |
                                                v
                    +-------------------------------------------------------+
                    |             ADAPTIVE MLOPS & RETRAINING               |
                    | Statistical Drift Monitoring (PSI / KS-Test)          |
                    | Quality Gate Promotion (Macro F1) | Rollback Engine   |
                    | Airflow Orchestration DAGs (Ingestion, Train, Drift)  |
                    +-------------------------------------------------------+
```

---

## 3. End-to-End ML Lifecycle & System Workflow

The complete lifecycle operates across 8 distinct phases:

1. **Ingestion:**  
   Daily incremental pulls fetch new CVE metadata from NIST NVD. The FIRST EPSS gzipped daily CSV snapshot is decompressed and appended into `epss_history`. The CISA KEV catalog is diffed against yesterday's snapshot and appended into `kev_history`.
2. **Validation:**  
   `DataValidator` enforces range boundaries (`cvss_score` ∈ [0,10], `epss_score` ∈ [0,1]), valid regex format (`CVE-\d{4}-\d{4,}`), and non-null constraints using Pandera schemas.
3. **Stage 1 Feature Extraction:**  
   `FeatureEngineer` converts raw CVSS v3.1 vector strings into 22 binary indicators (Attack Vector, Attack Complexity, Privileges Required, etc.), extracts CWE groupings, and mines description keywords. **EPSS, KEV flags, and CVSS scores are strictly excluded from Stage 1 features to eliminate target leakage.**
4. **Model Training & Experimentation:**  
   Data is split temporally based on `published_date`. `SeverityClassifier` applies SMOTE oversampling to balance minority severity classes (NONE, LOW), fits an XGBoost multi-class softprob model, and calibrates output probabilities using Isotonic Regression. All parameters, metrics (Macro F1, Critical Recall, ECE), and model artifacts are tracked in MLflow.
5. **Quality Gate & Model Promotion:**  
   `ModelQualityGate` evaluates challenger model metrics against production champion metrics. A challenger must exceed the champion's Macro F1 score by a defined margin (>= 0.01) to be promoted to Production in MLflow; otherwise, it is rejected.
6. **Stage 2 Risk Scoring:**  
   `RiskScorer` combines Stage 1 calibrated severity predictions with real-time threat intelligence (EPSS score, CISA KEV status) and asset context (asset criticality, network exposure) into a composite priority score (0.0 to 100.0) mapping to remediation SLA tiers:
   - **CRITICAL_RISK (80–100):** SLA 24–72 hours
   - **HIGH_RISK (60–79):** SLA 7 days
   - **MEDIUM_RISK (30–59):** SLA 30 days
   - **LOW_RISK (0–29):** Best effort
7. **Serving & Explainability:**  
   FastAPI exposes endpoints (`/vulnerabilities`, `/risk-score`, `/explain`, `/models`, `/drift`, `/assets`, `/remediation`). `/explain` executes `VulnerabilityExplainer` to generate SHAP feature contribution waterfall data for any given prediction.
8. **Drift Monitoring & Retraining:**  
   `StatisticalDriftDetector` calculates Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) test statistics comparing incoming inference feature distributions against baseline training distributions. When PSI exceeds 0.25 on key features, Airflow triggers the automated retraining DAG.

---

## 4. Work Completed Till Now

| Area | Component | Implementation Details | Status |
|---|---|---|---|
| **Database** | PostgreSQL Schema (`init.sql`) | Append-only `epss_history` and `kev_history` tables; `cves`, `asset_inventory`, `asset_cve_map`, `predictions_log`, `model_registry`, `drift_reports` tables initialized. | **COMPLETED** |
| **Ingestion** | Bulk Clients (`src/ingestion/`) | Hardened `NVDClient`, `EPSSClient` (bulk gzipped CSV parser), `CISAKEVClient` (catalog diffing), and `AssetMatcher`. | **COMPLETED** |
| **Validation** | Data Schema (`src/validation/`) | `DataValidator` and `VulnerabilitySchemaValidator` enforcing Pandera constraints with bad-row thresholds. | **COMPLETED** |
| **Features** | Feature Engineering (`src/features/`) | `CVSSVectorParser` parsing CVSS strings into binary metrics; `FeatureEngineer` generating text/CWE features. | **COMPLETED** |
| **ML Model** | Classifier & XAI (`src/models/`) | `SeverityClassifier` (XGBoost + SMOTE + Isotonic Calibration); `VulnerabilityExplainer` (SHAP waterfall values). | **COMPLETED** |
| **Risk Scorer** | Risk Engine (`src/models/risk_scorer.py`) | `RiskScorer` calculating multi-factor 0-100 risk score and SLA assignment. | **COMPLETED** |
| **API** | FastAPI App (`src/serving/api/`) | Core routes (`/predict`, `/risk-score`, `/explain`, `/health`) operational with schema verification. | **COMPLETED** |
| **Orchestration** | Airflow DAGs (`airflow/dags/`) | Rebuilt `data_ingestion_dag`, `model_training_dag`, and `drift_monitoring_dag` with correct module imports. | **COMPLETED** |
| **Pipeline Test** | E2E Runner (`scripts/run_pipeline.py`) | Automated end-to-end execution test running ingestion, feature engineering, model training, MLflow tracking, and risk scoring without errors. | **COMPLETED** |

---

## 5. Instructions for Team Members

### A. Instructions for Sanskar (ML & MLOps Lead)

Sanskar, your core focus is maintaining ML model rigor, ensuring zero data leakage, and finalizing the MLOps promotion/rollback loop.

#### 1. ML Target Lock & Baseline Documentation:
- Document the choice of Stage 1 target (CVSS Base Severity multi-class: NONE/LOW/MEDIUM/HIGH/CRITICAL) in the model documentation.
- Compare trained XGBoost performance against a trivial baseline (e.g. mapping CVSS vector severity directly).

#### 2. Implement Mandatory Leakage Guard Test (`tests/test_leakage_guard.py`):
Create an automated test asserting that forbidden post-publication columns (`epss_score`, `epss_percentile`, `kev_flag`, `cvss_score`, `cvss_severity`) are **NEVER** present in the Stage 1 feature matrix `build_features()`.

```python
# tests/test_leakage_guard.py
def test_no_forbidden_features_in_matrix():
    from src.features.feature_engineer import FeatureEngineer
    sample_raw = {
        "cvss_vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
        "description": "SQL injection vulnerability in auth module",
        "cwe": "CWE-89",
        "epss_score": 0.85,
        "kev_flag": True
    }
    features = FeatureEngineer().transform_row(sample_raw)
    forbidden = ['epss_score', 'epss_percentile', 'kev_flag', 'cvss_score', 'cvss_severity']
    for f in forbidden:
        assert f not in features, f"Target leakage detected! Feature '{f}' found in Stage 1 matrix."
```

#### 3. Align `RiskScorer` Weights & Worked-Example Assertion:
Ensure `src/models/risk_scorer.py` strictly uses the 5-component weight formula contract:
$$\text{Score} = (0.30 \cdot S_{\text{ML}} + 0.25 \cdot S_{\text{EPSS}} + 0.25 \cdot S_{\text{KEV}} + 0.15 \cdot S_{\text{Asset}} + 0.05 \cdot S_{\text{Exposure}}) \times 100$$
Add `tests/test_risk_formula.py` with an exact worked-example test:
```python
# tests/test_risk_formula.py
def test_risk_scorer_worked_example():
    from src.models.risk_scorer import RiskScorer
    scorer = RiskScorer()
    # Worked example: ml_prob=0.72, epss=0.55, kev=True (1.0), asset_crit=0.8, exposure=True (1.0)
    # Expected score = (0.30*0.72 + 0.25*0.55 + 0.25*1.0 + 0.15*0.8 + 0.05*1.0)*100 = 77.35
    result = scorer.calculate_risk(
        ml_severity_confidence=0.72,
        epss_score=0.55,
        is_kev=True,
        asset_criticality=0.8,
        is_internet_facing=True
    )
    assert abs(result.risk_score - 77.35) < 0.01, f"Expected 77.35, got {result.risk_score}"
```

#### 4. Model Promotion & Rollback Module (`src/mlops/promotion.py`):
Implement explicit champion/challenger promotion and rollback logic in `src/mlops/promotion.py` with accompanying unit tests in `tests/test_promotion_logic.py` verifying both `promoted` and `rejected` scenarios.

#### 5. Execute Experiments A & B:
- **Experiment A (Real Data Drift):** Execute drift monitoring on real ingested CVE data and log PSI/KS results into `drift_reports`.
- **Experiment B (Controlled Synthetic Perturbation):** Perturb a feature column in a validation batch (e.g. inflating network-facing attack vectors), run `drift_detector.py`, and verify that the drift alarm triggers.

---

### B. Instructions for Chinmayi (API & Frontend Lead)

Chinmayi, your focus is replacing all mock API endpoints with real database/MLflow queries and wiring the 5-page frontend web dashboard.

#### 1. Replace Mock Endpoints with Real Database Queries:
Update all route files in `src/serving/api/routes/` to query PostgreSQL and MLflow directly:
- **`vulnerabilities.py` (`GET /vulnerabilities`):** Return paginated rows from `cves` joined with latest `epss_history`, `kev_history`, and calculated risk scores. Must accept `limit`, `sort`, `kev_only`, and `min_cvss` parameters.
- **`models.py` (`GET /models`):** Query MLflow tracking server and `model_registry` table to return champion version, training date, and metrics.
- **`drift.py` (`GET /drift`):** Query `drift_reports` table to return the latest PSI/KS drift check results.
- **`assets.py` (`GET/POST/PUT /assets`):** Fetch asset inventory and trigger `AssetMatcher` recomputation on asset insert/update.
- **`remediation.py` (`GET /remediation`):** Fetch CVEs present in `asset_cve_map` ordered by `risk_score` descending.

#### 2. Implement API Route Verification Tests (`tests/test_api_routes.py`):
Create unit/integration tests verifying that endpoints return real dynamic data from PostgreSQL rather than static fixtures. For instance, query `/vulnerabilities`, insert a new CVE row into PostgreSQL, query `/vulnerabilities` again, and assert the count/contents change.

#### 3. Frontend Web Dashboard Architecture (5 Core Pages):
Build or wire the dashboard frontend to consume FastAPI endpoints dynamically (no hardcoded data arrays in UI code):
1. **Explorer Page (`/explorer`):** Server-side paginated table with CVSS/EPSS/KEV filters, Top-N dropdown, and row clicks opening a SHAP waterfall explanation modal (`GET /vulnerabilities/{id}/explanation`).
2. **Remediation Page (`/remediation`):** Prioritized list of vulnerabilities affecting enterprise assets in `asset_cve_map`, display of asset criticality, and SLA countdown badges.
3. **Assets Page (`/assets`):** Asset inventory management form (add/edit assets) triggering live recomputation of `asset_cve_map`.
4. **Monitoring Page (`/monitoring`):** Champion model performance card (Macro F1, ECE), live drift detection status table, MLflow run history, and a "Trigger Retrain" button (`POST /retrain`).
5. **CVE Timeline Page (`/cve/{id}`):** Historical trend line chart plotting EPSS score progression over time alongside KEV publication markers.

---

## 6. Definition of Done & Hand-Off Matrix

```
+-----------------------------------------------------------------------------------+
|                            SYSTEM HAND-OFF CONTRACT                               |
+-------------------+-----------------------------------+---------------------------+
| Lead              | Deliverable Provided              | Consumer / Hand-Off To    |
+-------------------+-----------------------------------+---------------------------+
| Shlok (Data/Infra)| Real PostgreSQL DB Schema         | Sanskar (for training)    |
|                   | Append-only EPSS/KEV History      | Chinmayi (for API queries)|
|                   | Clean Airflow DAGs                | System Orchestration      |
+-------------------+-----------------------------------+---------------------------+
| Sanskar (ML/MLOps)| Leakage-free Stage 1 Model        | Chinmayi (for API /explain|
|                   | Standard Risk Formula (0-100)     | Chinmayi (for API /risk)  |
|                   | PSI/KS Drift Engine               | Chinmayi (for API /drift) |
|                   | Quality Gate & MLflow Tracking    | Chinmayi (for API /models)|
+-------------------+-----------------------------------+---------------------------+
| Chinmayi (Serving)| Dynamic FastAPI Backend           | End Users & Evaluators    |
|                   | 5-Page Responsive Dashboard       | Executive Demonstration   |
|                   | Dynamic Integration Unit Tests    | Automated CI/CD Pipeline  |
+-------------------+-----------------------------------+---------------------------+
```

### Final Master Checklist:
- [x] PostgreSQL database running with append-only `epss_history` and `kev_history` tables.
- [x] Ingestion pipeline fetching real NVD, EPSS bulk CSV, and CISA KEV data.
- [x] Airflow DAGs synchronized with real Python functions.
- [ ] `tests/test_leakage_guard.py` passing zero-leakage assertions.
- [ ] `tests/test_risk_formula.py` verifying exact 77.35 worked example.
- [ ] Champion/Challenger promotion and rollback module (`promotion.py`) tested.
- [ ] Experiments A (real drift) and B (synthetic perturbation) executed.
- [ ] FastAPI endpoints returning live DB and MLflow query results.
- [ ] 5-page web dashboard operational without hardcoded frontend arrays.
- [ ] Complete E2E verification test passing in CI/CD pipeline.
