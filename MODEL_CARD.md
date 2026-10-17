# Model Card: Software Vulnerability Severity & Risk Prioritization Classifier

> **Analogy**: *The nutrition label on a food packet — declaring exactly what ingredients went into the model, what it delivers, its known blind spots, and safe operating bounds.*

---

## 1. Model Details
- **Model Name**: Vulnerability Severity & Risk Prioritizer (`vulnerability-severity-classifier`)
- **Version**: 2.0 (Symptom-Based, Leakage-Free)
- **Model Type**: Calibrated Gradient Boosted Ensemble (XGBoost + Isotonic Probability Calibration) & CISA KEV Forecaster
- **Maintainer**: Adaptive MLOps Core Platform Team
- **Release Date**: October 2026
- **License**: Apache 2.0
- **Registry Aliases**: `@champion` (Production inference), `@challenger` (Shadow evaluation)

---

## 2. Intended Use & Application Scope
- **Primary Task**: Predict vulnerability base severity (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`) immediately upon initial disclosure using strictly early symptom indicators (CVE description text, CWE weakness IDs, affected vendor/product context, and reference links).
- **Secondary (Bonus) Task**: Forecast probability of active in-the-wild exploitation within 30 to 90 days (`CISA KEV` listing prediction).
- **Composite Risk Scoring**: Prioritize organizational remediation using a weighted composite risk formula combining ML predicted severity, EPSS exploitation probability, CISA KEV presence, asset business criticality, and network exposure.
- **Out-of-Scope / Misuse**:
  - Do NOT use as a replacement for human security code audits or penetration testing.
  - Do NOT deploy on non-CVE unstructured vulnerability text (e.g., raw decompiled binaries or bug tracker tickets without descriptions).

---

## 3. Training & Validation Data
- **Data Source**: Official National Vulnerability Database (NVD API 2.0) records spanning 2019 to 2026, cross-referenced with CISA Known Exploited Vulnerabilities (KEV) Catalog.
- **Dataset Size**: 7,157 real CVE records (5,815 with verified official ground truth labels).
- **Class Distribution**:
  - `HIGH`: 2,816 (~48.4%)
  - `MEDIUM`: 2,097 (~36.1%)
  - `CRITICAL`: 670 (~11.5%)
  - `LOW`: 232 (~4.0%)
- **Data Preprocessing & Splitting**:
  - **Temporal Split**: Chronological ordering by `published_date` (older CVEs in train set, newer CVEs in validation and test sets) to prevent future-data leakage.
  - **Zero Leakage Enforcement**: CVSS vector submetrics (`AV`, `AC`, `PR`, `UI`, `S`, `C`, `I`, `A`), `cvss_score`, `epss_score`, and `in_cisa_kev` are strictly prohibited as features for severity prediction.
  - **Features**: 100-dimensional TF-IDF description n-grams, 10 security keyword indicators (RCE, SQLi, Buffer Overflow, Auth Bypass, Priv Esc, DoS, etc.), CWE weakness categories, and top vendor one-hot flags.

---

## 4. Performance & Baseline Benchmarking
Evaluated on the held-out temporal test set:

| Model Hierarchy | Macro F1 | Weighted F1 | Critical Recall | High Recall | Calibration ECE |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Always-Guess-Most-Common** | 0.1681 | 0.3404 | 0.0000 | 1.0000 | 0.0000 |
| **2. Keyword Rules Baseline** | 0.1441 | 0.1488 | 0.5400 | 0.1131 | 0.7125 |
| **3. Logistic Regression (Class Weights)** | 0.2019 | 0.2542 | 0.3800 | 0.1109 | 0.2439 |
| **4. XGBoost (Calibrated)** | **0.2970** | **0.4392** | **0.2400** | **0.2964** | **0.1291** |
| **5. LightGBM (Class Weights)** | **0.3082** | **0.4496** | **0.2400** | **0.3190** | **0.1468** |

### Class Imbalance Handling
- **SMOTE vs Class Weights**: SMOTE on text coordinates interpolates between sparse word vectors, creating synthetic artifacts with high noise. Cost-sensitive Class Weights (`compute_sample_weight('balanced')`) preserve real token vectors and achieve double the recall on minority `CRITICAL` vulnerabilities.

---

## 5. Live Inference Latency Profile
Measured on FastAPI production runtime:
- **Throughput**: 11.20 Requests/Second
- **Latency p50**: **14.59 ms**
- **Latency p90**: **17.14 ms**
- **Latency p95**: **19.14 ms**
- **Latency p99**: **100.81 ms**

---

## 6. Known Failure Modes & Limitations
1. **Short / Cryptic Advisory Descriptions**:
   - CVEs with vague descriptions (e.g., *"An issue was discovered in Product X"*) lack semantic cues for attack mechanics, leading the model to default to `HIGH` (the empirical prior).
2. **Multi-Step Exploit Chains**:
   - High-severity chains requiring multiple chained low-severity bugs cannot be inferred if each individual CVE description only mentions the isolated component flaw.
3. **Novel Attack Terminology**:
   - Zero-day novel attack concepts using vocabulary not present in the historical NVD training set may trigger embedding distribution drift.

---

## 7. MLOps Safeguards & Operational Monitoring
- **Prediction Auditing**: 100% of `/predict` and `/risk-score` requests are recorded in PostgreSQL `predictions_log`.
- **Text & Embedding Drift**: Monitored via TF-IDF centroid cosine distance and CWE Chi-Square distribution testing every 6 hours in Airflow.
- **Late Ground Truth Calibration**: Tracks newly published NVD CVSS scores against past model predictions to compute live Macro F1.
- **Human-in-the-Loop Feedback**: Analyst overrides submitted to `/api/v1/feedback` are logged to `analyst_feedback` and automatically staged for scheduled retraining.
