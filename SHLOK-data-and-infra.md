# SHLOK — Data Engineering & Backend Infrastructure

> Your scope: everything data touches before the ML model sees it, plus the
> infra that keeps the whole system running. This is the foundation — Sanskar's
> ML work and Chinmayi's frontend both depend on the DB schema and ingestion
> pipeline you build here. Lock the schema early and communicate changes to both.

---

## 0. Why we're rebuilding this part of the friend's project

His ingestion clients (NVD/EPSS/KEV) are decent and reusable. Everything
downstream of them — the synthetic data generator, the Airflow DAGs, the mock
API responses — is broken or fake. Your job is to make the *real* data flow
correctly from API → DB → feature-ready state, and to make the orchestration
around it actually run without errors, unlike his version.

**The one rule that governs everything you build:** never overwrite historical
EPSS/KEV values. Every score/status change over time must be a new row, not an
update. This is what makes leakage-free ML (Sanskar's job) and honest drift
detection possible. If you get this wrong, everything downstream breaks silently.

---

## 1. Data sources — what you're pulling, and how

| Source | What | How | Cost |
|---|---|---|---|
| NVD API | CVE details, CVSS, CWE, affected products | Incremental pull via `pubStartDate`/`lastModDate` + pagination | Free (get an API key for higher rate limits) |
| FIRST EPSS | Daily exploitation probability per CVE | Daily bulk CSV: `epss.empiricalsecurity.com/epss_scores-current.csv.gz` — use this, NOT the per-CVE lookup API | Free, no registration |
| CISA KEV | Confirmed-exploited catalog | Full JSON/CSV pull, diffed vs. yesterday's snapshot | Free |

Reuse and harden: `src/ingestion/nvd_client.py`, `epss_client.py`,
`cisa_kev_client.py` from the friend's repo. Before trusting them, actually
run each one and verify: pagination works, rate-limit backoff works, no silent
failures on bad responses.

## 2. Database schema (own this — this is the shared contract)

```
cves               (cve_id PK, published_date, cvss_vector, cvss_score,
                     cwe, description, vendor, product, ...)
epss_history       (cve_id, score, percentile, snapshot_date)  -- APPEND ONLY
kev_history        (cve_id, added_date, first_seen_in_kev)     -- APPEND ONLY
asset_inventory    (asset_id PK, product, version, criticality, internet_facing)
asset_cve_map      (asset_id, cve_id)  -- computed by matching product/version vs NVD CPE data
predictions_log    (cve_id, model_version, predicted_at, prediction, features_snapshot)
model_registry     (model_version, trained_at, metrics_json, status)  -- Sanskar writes to this, you own the table
drift_reports      (run_date, feature, metric, value, threshold, flagged)  -- Sanskar writes to this, you own the table
```

Set this up in PostgreSQL. Share the exact schema (with column types) with
Sanskar and Chinmayi before either of them starts writing code against it.

## 3. Daily ingestion pipeline (your core deliverable)

1. **NVD incremental fetch** — new/modified CVEs since last successful sync
2. **EPSS daily bulk download** — decompress CSV, append one row per CVE per
   day into `epss_history` (never overwrite)
3. **KEV catalog refresh** — fetch current catalog, diff vs. yesterday, insert
   new entries into `kev_history`
4. **Trigger downstream recompute** — after ingestion, signal that risk scores
   need recalculating (Sanskar's risk engine consumes this)

## 4. Historical backfill (kill the synthetic dataset)

The friend's `seed_data.py` generates fake data from 8 copy-pasted templates
with random dates — this is why his model looked artificially accurate and
why his temporal validation was meaningless (same templates on both sides of
the train/test split). **Delete this from anything used for training/metrics.**

Instead:
- Pull real NVD CVEs across a multi-year window (e.g. 2019–2026)
- Pull FIRST's historical EPSS archive (available back to April 2021) to
  reconstruct time-stamped EPSS values
- Pull the full KEV catalog with `dateAdded` for exploitation ground truth
- Join by CVE ID respecting timestamps — hand this real, joined dataset to Sanskar

## 5. Data validation rules

- `cvss_score` in [0, 10]
- `epss_score` in [0, 1]
- `cve_id` matches `CVE-\d{4}-\d{4,}`
- `published_date` not null, not in the future
- Fail (don't silently continue) if a batch exceeds a bad-row threshold; log which rule failed and the count

## 6. Simulated asset inventory

Build a small fixture: fictional company's software (product, version,
criticality 1–5, internet-facing bool). Match against NVD affected-product/CPE
data to populate `asset_cve_map`. Document clearly that this inventory is
simulated — the CVE/EPSS/KEV data behind it is real.

## 7. Airflow — rebuild, don't patch

The friend's DAGs call functions that don't exist in the real modules:
- DAG calls `fetch_recent_cves()` → real function is `fetch_cves_since()`
- DAG calls `fetch_catalog()` → real function is `fetch()`
- Import references `VulnerabilitySchemaValidator` → real class is `DataValidator`
- Reference to a missing `features.pipeline` module

**Before wiring any DAG task, import the function yourself and confirm it
exists with that exact name and signature.** Don't assume from old docs.

Build three separate DAGs:
1. Daily ingestion (steps in section 3)
2. Drift monitoring (runs right after ingestion — mostly Sanskar's logic, but
   you own the DAG scheduling it)
3. Retraining (triggered by drift alert or schedule — mostly Sanskar's logic,
   same deal)

## 8. Infrastructure you own

- **Docker Compose**: postgres, airflow-webserver, airflow-scheduler,
  airflow-worker, mlflow, api, frontend — reuse the friend's `docker-compose.yml`
  as a base but verify every service actually starts and connects
- **CI/CD**: lint → unit tests → integration test (spin up docker-compose,
  run one ingestion cycle against a fixture, assert DB populated correctly) →
  build images. Reuse `.github/workflows/ci_cd.yml` as a base, extend it
- **Config management**: one `.env.example` for DB creds, NVD API key, MLflow
  URI; one `config.yaml` for thresholds (used across all three of you — don't
  hardcode values that Sanskar or Chinmayi also need)
- **Data versioning**: DVC or a content-hash + stored snapshot per ingestion
  batch, so any model run can be traced back to exact input data

## 9. Your Definition of Done

- [ ] Real historical dataset in Postgres, validated, documented row count/date range, zero synthetic rows
- [ ] `epss_history`/`kev_history` are append-only and verified to never overwrite
- [ ] All 3 Airflow DAGs run end-to-end locally without task failures
- [ ] Asset inventory + `asset_cve_map` matching works and is queryable
- [ ] Docker Compose brings up all services with one command, no manual fixes
- [ ] CI pipeline passes, including the integration test against a fixture dataset
- [ ] Schema and API contract shared with Sanskar and Chinmayi, versioned in the repo

## 10. Handoff points

- **To Sanskar:** the joined historical dataset (CVE + time-stamped EPSS/KEV),
  the DB schema, and the config file for thresholds
- **To Chinmayi:** the DB schema (so her FastAPI routes query correctly) and
  confirmation of what real data looks like (so she's not building against
  mock shapes)
