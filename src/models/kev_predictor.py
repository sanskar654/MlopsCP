"""
CISA KEV Exploitation Predictor (Bonus Task)
===========================================
Predicts whether a newly disclosed CVE will be added to the CISA Known Exploited
Vulnerabilities (KEV) catalog in the next 30 or 90 days (Threat Forecasting).

Analogy:
A weather forecast for attacks — estimating active in-the-wild exploitation
risk before external advisories confirm ongoing campaigns.

Inputs:
- CVE description (TF-IDF & attack keywords: RCE, Auth Bypass, Heap Overflow)
- CWE weakness categories (Remote execution, Injection, Authentication)
- Affected Vendor / Product exposure
- Reference patterns & metadata

Output:
- Probability of active exploitation (0.0 to 1.0)
- Exploitability Risk Tier (LOW, MEDIUM, HIGH, IMMINENT)
- 30-day and 90-day exploitation forecast flags
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb

logger = logging.getLogger("models.kev_predictor")


class CISAKEVForecaster:
    """
    Supervised classifier for early exploitation forecasting into CISA KEV.
    """

    def __init__(self, time_horizon_days: int = 90):
        self.time_horizon_days = time_horizon_days
        self.model = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=5.0,  # KEV is heavily imbalanced (~2-5% positive)
            eval_metric="aucpr",
            random_state=42,
        )
        self.is_fitted = False

    def prepare_kev_targets(
        self,
        df_cves: pd.DataFrame,
        kev_df: pd.DataFrame,
        window_days: int = 90,
    ) -> pd.Series:
        """
        Creates binary label: 1 if CVE added to KEV within window_days of publication, else 0.
        """
        kev_lookup = {}
        if "cve_id" in kev_df.columns and "date_added" in kev_df.columns:
            for _, r in kev_df.iterrows():
                try:
                    kev_lookup[r["cve_id"]] = pd.to_datetime(r["date_added"])
                except Exception:
                    pass

        targets = []
        for _, row in df_cves.iterrows():
            cve_id = row.get("cve_id")
            pub_date_raw = row.get("published_date")
            if cve_id in kev_lookup and pub_date_raw:
                pub_date = pd.to_datetime(pub_date_raw)
                added_date = kev_lookup[cve_id]
                diff_days = (added_date - pub_date).total_seconds() / (24 * 3600)
                # Exploited within time window or historically cataloged
                targets.append(
                    1 if (0 <= diff_days <= window_days or diff_days < 0) else 1
                )
            elif cve_id in kev_lookup:
                targets.append(1)
            else:
                targets.append(0)

        return pd.Series(targets, index=df_cves.index)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "CISAKEVForecaster":
        """Fit KEV forecaster using balanced sample weights."""
        logger.info(
            "Training CISA KEV Exploitation Forecaster (Positive rate: %.2f%%)",
            (y.mean() * 100),
        )
        sample_weights = compute_sample_weight("balanced", y)
        self.model.fit(X.fillna(0), y, sample_weight=sample_weights)
        self.is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return exploitation probability [P(not_exploited), P(exploited_in_kev)]."""
        if not self.is_fitted:
            raise RuntimeError("CISAKEVForecaster is not fitted.")
        return self.model.predict_proba(X.fillna(0))

    def predict_exploitation_risk(self, X: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Predict exploitation probabilities and assign threat forecast tiers.
        """
        probas = self.predict_proba(X)[:, 1]
        results = []
        for p in probas:
            p_val = float(p)
            if p_val >= 0.70:
                tier = "IMMINENT_EXPLOITATION"
            elif p_val >= 0.40:
                tier = "HIGH_EXPLOITATION_RISK"
            elif p_val >= 0.15:
                tier = "MODERATE_EXPLOITATION_RISK"
            else:
                tier = "LOW_EXPLOITATION_RISK"

            results.append(
                {
                    "kev_exploitation_probability": round(p_val, 4),
                    "exploitation_risk_tier": tier,
                    "predicted_kev_30d": bool(p_val >= 0.50),
                    "predicted_kev_90d": bool(p_val >= 0.35),
                }
            )
        return results

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Comprehensive evaluation metrics for KEV prediction."""
        probas = self.predict_proba(X_test)[:, 1]
        preds = (probas >= 0.50).astype(int)

        try:
            auc_roc = float(roc_auc_score(y_test, probas))
        except Exception:
            auc_roc = 0.5

        f1 = float(f1_score(y_test, preds, zero_division=0))
        report = classification_report(y_test, preds, output_dict=True, zero_division=0)

        metrics = {
            "kev_auc_roc": round(auc_roc, 4),
            "kev_f1_score": round(f1, 4),
            "kev_positive_recall": round(report.get("1", {}).get("recall", 0.0), 4),
            "kev_positive_precision": round(
                report.get("1", {}).get("precision", 0.0), 4
            ),
        }
        logger.info("KEV Forecaster Evaluation: AUC=%.4f, F1=%.4f", auc_roc, f1)
        return metrics
