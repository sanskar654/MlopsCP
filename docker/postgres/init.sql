-- =============================================================================
-- PostgreSQL Initialization Script
-- Vulnerability MLOps Platform (Data Engineering & Infrastructure)
-- =============================================================================

-- ───────────────────────────────────────────────
-- 1. Primary CVE Table
-- ───────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS cves (
    cve_id              VARCHAR(20) PRIMARY KEY,
    published_date      TIMESTAMPTZ NOT NULL,
    last_modified_date  TIMESTAMPTZ,
    cvss_vector         TEXT,
    cvss_score          NUMERIC(4,1),
    base_severity       VARCHAR(10),
    cwe                 TEXT,
    description         TEXT,
    vendor              VARCHAR(255),
    product             VARCHAR(255),
    raw_json            JSONB,
    ingested_at         TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_cves_published ON cves(published_date);
CREATE INDEX IF NOT EXISTS idx_cves_severity ON cves(base_severity);

-- Legacy Raw Vulnerabilities table for backwards compatibility
CREATE TABLE IF NOT EXISTS raw_vulnerabilities (
    id                  SERIAL PRIMARY KEY,
    cve_id              VARCHAR(20) NOT NULL UNIQUE,
    published_date      TIMESTAMPTZ NOT NULL,
    last_modified_date  TIMESTAMPTZ,
    vuln_status         VARCHAR(50),
    description         TEXT,
    cvss_version        VARCHAR(10),
    cvss_vector_string  TEXT,
    attack_vector       VARCHAR(20),
    attack_complexity   VARCHAR(10),
    privileges_required VARCHAR(10),
    user_interaction    VARCHAR(10),
    scope               VARCHAR(20),
    confidentiality_impact VARCHAR(10),
    integrity_impact    VARCHAR(10),
    availability_impact VARCHAR(10),
    base_score          NUMERIC(4,1),
    base_severity       VARCHAR(10),
    exploitability_score NUMERIC(4,1),
    impact_score        NUMERIC(4,1),
    cvss_v2_score       NUMERIC(4,1),
    cvss_v2_severity    VARCHAR(10),
    cwe_ids             TEXT[],
    reference_urls      TEXT[],
    cpe_list            TEXT[],
    vendor_count        INTEGER DEFAULT 0,
    reference_count     INTEGER DEFAULT 0,
    ingested_at         TIMESTAMPTZ DEFAULT NOW(),
    data_source         VARCHAR(20) DEFAULT 'NVD',
    schema_version      VARCHAR(10) DEFAULT '2.0'
);

CREATE INDEX IF NOT EXISTS idx_raw_vulns_cve_id ON raw_vulnerabilities(cve_id);

-- ───────────────────────────────────────────────
-- 2. EPSS History (APPEND-ONLY, Never Overwrite)
-- ───────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS epss_history (
    id             SERIAL PRIMARY KEY,
    cve_id         VARCHAR(20) NOT NULL,
    score          NUMERIC(8,6) NOT NULL,
    percentile     NUMERIC(8,6) NOT NULL,
    snapshot_date  DATE NOT NULL,
    ingested_at    TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT epss_hist_unique_cve_date UNIQUE(cve_id, snapshot_date)
);

CREATE INDEX IF NOT EXISTS idx_epss_hist_cve ON epss_history(cve_id);
CREATE INDEX IF NOT EXISTS idx_epss_hist_date ON epss_history(snapshot_date);
CREATE INDEX IF NOT EXISTS idx_epss_hist_score ON epss_history(score DESC);

-- Legacy epss_scores table for backwards compatibility
CREATE TABLE IF NOT EXISTS epss_scores (
    id          SERIAL PRIMARY KEY,
    cve_id      VARCHAR(20) NOT NULL,
    score_date  DATE NOT NULL,
    epss_score  NUMERIC(8,6) NOT NULL,
    percentile  NUMERIC(8,6) NOT NULL,
    ingested_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(cve_id, score_date)
);

-- ───────────────────────────────────────────────
-- 3. KEV History (APPEND-ONLY, Never Overwrite)
-- ───────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS kev_history (
    id                 SERIAL PRIMARY KEY,
    cve_id             VARCHAR(20) NOT NULL,
    added_date         DATE NOT NULL,
    first_seen_in_kev  TIMESTAMPTZ DEFAULT NOW(),
    vendor_project     VARCHAR(255),
    product            VARCHAR(255),
    vulnerability_name TEXT,
    short_description  TEXT,
    required_action    TEXT,
    due_date           DATE,
    CONSTRAINT kev_hist_unique_cve_date UNIQUE(cve_id, added_date)
);

CREATE INDEX IF NOT EXISTS idx_kev_hist_cve ON kev_history(cve_id);
CREATE INDEX IF NOT EXISTS idx_kev_hist_added ON kev_history(added_date);

-- Legacy cisa_kev table for backwards compatibility
CREATE TABLE IF NOT EXISTS cisa_kev (
    id               SERIAL PRIMARY KEY,
    cve_id           VARCHAR(20) NOT NULL UNIQUE,
    vendor_project   VARCHAR(255),
    product          VARCHAR(255),
    vulnerability_name TEXT,
    date_added       DATE NOT NULL,
    short_description TEXT,
    required_action  TEXT,
    due_date         DATE,
    known_ransomware_campaign_use VARCHAR(10) DEFAULT 'Unknown',
    notes            TEXT,
    ingested_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ───────────────────────────────────────────────
-- 4. Asset Inventory & Asset-CVE Mapping
-- ───────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS asset_inventory (
    asset_id        VARCHAR(100) PRIMARY KEY,
    product         VARCHAR(255) NOT NULL,
    version         VARCHAR(100) NOT NULL,
    criticality     INTEGER CHECK (criticality BETWEEN 1 AND 5),
    internet_facing BOOLEAN DEFAULT FALSE,
    asset_name      VARCHAR(255),
    asset_type      VARCHAR(50),
    owner           VARCHAR(100),
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS asset_cve_map (
    asset_id     VARCHAR(100) NOT NULL REFERENCES asset_inventory(asset_id) ON DELETE CASCADE,
    cve_id       VARCHAR(20) NOT NULL,
    matched_at   TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (asset_id, cve_id)
);

CREATE INDEX IF NOT EXISTS idx_asset_cve_map_cve ON asset_cve_map(cve_id);

-- ───────────────────────────────────────────────
-- 5. ML Serving & Logging Tables
-- ───────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS predictions_log (
    id                 SERIAL PRIMARY KEY,
    cve_id             VARCHAR(20) NOT NULL,
    model_version      VARCHAR(50) NOT NULL,
    predicted_at       TIMESTAMPTZ DEFAULT NOW(),
    prediction         VARCHAR(20) NOT NULL,
    confidence         NUMERIC(5,4),
    features_snapshot  JSONB
);

CREATE INDEX IF NOT EXISTS idx_pred_log_cve ON predictions_log(cve_id);

CREATE TABLE IF NOT EXISTS model_registry (
    id              SERIAL PRIMARY KEY,
    model_name      VARCHAR(100) DEFAULT 'vulnerability-severity-classifier',
    model_version   VARCHAR(20) NOT NULL UNIQUE,
    trained_at      TIMESTAMPTZ DEFAULT NOW(),
    metrics_json    JSONB,
    status          VARCHAR(20) DEFAULT 'challenger',
    mlflow_run_id   VARCHAR(100),
    model_role      VARCHAR(20) DEFAULT 'challenger',
    macro_f1        NUMERIC(6,4),
    critical_recall NUMERIC(6,4),
    ece             NUMERIC(6,4),
    roc_auc         NUMERIC(6,4),
    latency_p99_ms  NUMERIC(8,2),
    training_date   TIMESTAMPTZ,
    promoted_at     TIMESTAMPTZ,
    retired_at      TIMESTAMPTZ,
    promotion_notes TEXT
);

CREATE TABLE IF NOT EXISTS drift_reports (
    id                  SERIAL PRIMARY KEY,
    run_date            DATE NOT NULL DEFAULT CURRENT_DATE,
    feature             VARCHAR(100),
    metric              VARCHAR(50),
    value               NUMERIC(8,4),
    threshold           NUMERIC(8,4),
    flagged             BOOLEAN DEFAULT FALSE,
    report_type         VARCHAR(20) DEFAULT 'data_drift',
    dataset_drift       BOOLEAN DEFAULT FALSE,
    drift_score         NUMERIC(6,4),
    n_drifted_features INTEGER DEFAULT 0,
    drifted_features    TEXT[],
    report_path         TEXT,
    trigger_retrain     BOOLEAN DEFAULT FALSE,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_drift_reports_date ON drift_reports(run_date DESC);

-- ───────────────────────────────────────────────
-- 6. Engineered Features & Risk Scores
-- ───────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS engineered_features (
    id                     SERIAL PRIMARY KEY,
    cve_id                 VARCHAR(20) NOT NULL UNIQUE,
    feature_version        VARCHAR(20) DEFAULT '1.0',
    av_network             BOOLEAN DEFAULT FALSE,
    av_adjacent            BOOLEAN DEFAULT FALSE,
    av_local               BOOLEAN DEFAULT FALSE,
    av_physical            BOOLEAN DEFAULT FALSE,
    ac_low                 BOOLEAN DEFAULT FALSE,
    ac_high                BOOLEAN DEFAULT FALSE,
    pr_none                BOOLEAN DEFAULT FALSE,
    pr_low                 BOOLEAN DEFAULT FALSE,
    pr_high                BOOLEAN DEFAULT FALSE,
    ui_none                BOOLEAN DEFAULT FALSE,
    ui_required            BOOLEAN DEFAULT FALSE,
    scope_changed          BOOLEAN DEFAULT FALSE,
    ci_none                BOOLEAN DEFAULT FALSE,
    ci_low                 BOOLEAN DEFAULT FALSE,
    ci_high                BOOLEAN DEFAULT FALSE,
    ii_none                BOOLEAN DEFAULT FALSE,
    ii_low                 BOOLEAN DEFAULT FALSE,
    ii_high                BOOLEAN DEFAULT FALSE,
    ai_none                BOOLEAN DEFAULT FALSE,
    ai_low                 BOOLEAN DEFAULT FALSE,
    ai_high                BOOLEAN DEFAULT FALSE,
    days_since_published   INTEGER,
    quarter_published      INTEGER,
    year_published         INTEGER,
    is_recent              BOOLEAN DEFAULT FALSE,
    in_cisa_kev            BOOLEAN DEFAULT FALSE,
    epss_score             NUMERIC(8,6) DEFAULT 0.0,
    epss_percentile        NUMERIC(8,6) DEFAULT 0.0,
    has_exploit_reference  BOOLEAN DEFAULT FALSE,
    days_to_kev_listing    INTEGER,
    description_length     INTEGER DEFAULT 0,
    description_word_count INTEGER DEFAULT 0,
    has_rce_keyword        BOOLEAN DEFAULT FALSE,
    has_sqli_keyword       BOOLEAN DEFAULT FALSE,
    has_xss_keyword        BOOLEAN DEFAULT FALSE,
    has_overflow_keyword   BOOLEAN DEFAULT FALSE,
    has_auth_bypass_keyword BOOLEAN DEFAULT FALSE,
    has_priv_esc_keyword   BOOLEAN DEFAULT FALSE,
    has_dos_keyword        BOOLEAN DEFAULT FALSE,
    is_memory_safety_cwe   BOOLEAN DEFAULT FALSE,
    is_injection_cwe       BOOLEAN DEFAULT FALSE,
    is_auth_cwe            BOOLEAN DEFAULT FALSE,
    is_config_cwe          BOOLEAN DEFAULT FALSE,
    cwe_category_id        INTEGER,
    vendor_count           INTEGER DEFAULT 0,
    reference_count        INTEGER DEFAULT 0,
    has_patch_reference    BOOLEAN DEFAULT FALSE,
    severity_label         VARCHAR(10),
    severity_numeric       INTEGER,
    feature_created_at     TIMESTAMPTZ DEFAULT NOW(),
    feature_updated_at     TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS risk_scores (
    id                    SERIAL PRIMARY KEY,
    cve_id                VARCHAR(20) NOT NULL,
    computed_at           TIMESTAMPTZ DEFAULT NOW(),
    model_version         VARCHAR(50),
    predicted_severity    VARCHAR(10),
    predicted_severity_num INTEGER,
    model_confidence      NUMERIC(5,4),
    cvss_component        NUMERIC(5,4),
    kev_component         NUMERIC(5,4),
    epss_component        NUMERIC(5,4),
    asset_criticality     NUMERIC(5,4) DEFAULT 0.5,
    exposure_score        NUMERIC(5,4) DEFAULT 0.5,
    confidence_component  NUMERIC(5,4),
    risk_score            NUMERIC(6,2),
    risk_tier             VARCHAR(20),
    weight_config_version VARCHAR(20),
    UNIQUE(cve_id, computed_at)
);

CREATE TABLE IF NOT EXISTS analyst_feedback (
    id                  SERIAL PRIMARY KEY,
    cve_id              VARCHAR(20) NOT NULL,
    analyst_id          VARCHAR(100) NOT NULL,
    predicted_severity  VARCHAR(10),
    corrected_severity  VARCHAR(10) NOT NULL,
    feedback_notes      TEXT,
    confidence_rating   INTEGER CHECK (confidence_rating BETWEEN 1 AND 5),
    used_for_retraining BOOLEAN DEFAULT FALSE,
    submitted_at        TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS ingestion_log (
    id               SERIAL PRIMARY KEY,
    source           VARCHAR(20) NOT NULL,
    run_id           VARCHAR(100),
    started_at       TIMESTAMPTZ NOT NULL,
    completed_at     TIMESTAMPTZ,
    records_fetched  INTEGER DEFAULT 0,
    records_inserted INTEGER DEFAULT 0,
    records_updated  INTEGER DEFAULT 0,
    status           VARCHAR(20) DEFAULT 'running',
    error_message    TEXT,
    params           JSONB
);
