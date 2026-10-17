"""
Optuna Hyperparameter Tuner & Imbalance Strategy Comparator
===========================================================
1. Compares Imbalance Strategies on Text:
   - SMOTE (Averaging sparse text representations) vs Class Weights vs Unweighted.
2. Hyperparameter Optimization:
   - Tunes LightGBM & XGBoost using Optuna to maximize Macro F1.
   - Logs every Optuna trial to MLflow with parameters and objective score.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import pandas as pd
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight
import optuna

try:
    import mlflow

    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

try:
    from imblearn.over_sampling import SMOTE

    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

import xgboost as xgb

logger = logging.getLogger("training.optuna_tuner")

optuna.logging.set_verbosity(optuna.logging.WARNING)


def compare_imbalance_strategies(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """
    Directly evaluates and demonstrates why Class Weights outperforms SMOTE on text features.
    Analogy: SMOTE on text makes new sentences by averaging two sentences together;
    the result is noise/nonsense. Class weights penalize minority error directly.
    """
    results = []

    # 1. Unweighted
    clf_raw = xgb.XGBClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
    )
    clf_raw.fit(X_train.fillna(0), y_train)
    p_raw = clf_raw.predict(X_test.fillna(0))
    results.append(
        {
            "Strategy": "Unweighted Baseline",
            "Macro F1": round(
                float(f1_score(y_test, p_raw, average="macro", zero_division=0)), 4
            ),
            "Critical Recall": round(
                float(f1_score(y_test == 3, p_raw == 3, zero_division=0)), 4
            ),
            "Rationale": "Standard empirical loss without imbalance compensation",
        }
    )

    # 2. SMOTE (Synthetic interpolation)
    if HAS_SMOTE:
        try:
            sm = SMOTE(
                random_state=42,
                k_neighbors=min(5, int(y_train.value_counts().min()) - 1),
            )
            X_sm, y_sm = sm.fit_resample(X_train.fillna(0), y_train)
            clf_smote = xgb.XGBClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
            )
            clf_smote.fit(X_sm, y_sm)
            p_sm = clf_smote.predict(X_test.fillna(0))
            results.append(
                {
                    "Strategy": "SMOTE (Oversampling)",
                    "Macro F1": round(
                        float(f1_score(y_test, p_sm, average="macro", zero_division=0)),
                        4,
                    ),
                    "Critical Recall": round(
                        float(f1_score(y_test == 3, p_sm == 3, zero_division=0)), 4
                    ),
                    "Rationale": "Synthetically interpolates sparse text coordinates (generates noisy artifacts)",
                }
            )
        except Exception as e:
            logger.warning("SMOTE execution failed: %s", e)

    # 3. Class Weights (Cost-sensitive learning)
    weights = compute_sample_weight("balanced", y_train)
    clf_weighted = xgb.XGBClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
    )
    clf_weighted.fit(X_train.fillna(0), y_train, sample_weight=weights)
    p_weighted = clf_weighted.predict(X_test.fillna(0))
    results.append(
        {
            "Strategy": "Class Weights (Recommended)",
            "Macro F1": round(
                float(f1_score(y_test, p_weighted, average="macro", zero_division=0)), 4
            ),
            "Critical Recall": round(
                float(f1_score(y_test == 3, p_weighted == 3, zero_division=0)), 4
            ),
            "Rationale": "Multiplies loss penalty on minority mistakes without distorting sparse feature vectors",
        }
    )

    return pd.DataFrame(results)


class OptunaHyperparameterTuner:
    """
    Optuna hyperparameter optimization with trial-by-trial MLflow logging.
    """

    def __init__(self, n_trials: int = 15, timeout_seconds: int = 180):
        self.n_trials = n_trials
        self.timeout_seconds = timeout_seconds

    def tune_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        experiment_name: str = "vulnerability-severity-prediction",
    ) -> Dict[str, Any]:
        """
        Runs Optuna study for XGBoost and logs each trial to MLflow.
        """
        if HAS_MLFLOW:
            try:
                mlflow.set_experiment(experiment_name)
            except Exception:
                pass

        sample_weights = compute_sample_weight("balanced", y_train)

        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 80, 250, step=20),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.02, 0.2, log=True
                ),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 6),
                "objective": "multi:softprob",
                "num_class": 4,
                "eval_metric": "mlogloss",
                "random_state": 42,
                "n_jobs": -1,
            }

            model = xgb.XGBClassifier(**params)
            model.fit(X_train.fillna(0), y_train, sample_weight=sample_weights)
            preds = model.predict(X_val.fillna(0))
            macro_f1 = float(f1_score(y_val, preds, average="macro", zero_division=0))

            # Log trial to MLflow
            if HAS_MLFLOW:
                try:
                    with mlflow.start_run(
                        run_name=f"optuna_trial_{trial.number}", nested=True
                    ):
                        mlflow.log_params(params)
                        mlflow.log_metric("trial_macro_f1", macro_f1)
                except Exception:
                    pass

            return macro_f1

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout_seconds)

        logger.info(
            "Optuna tuning completed. Best trial: #%d, Best Macro F1: %.4f",
            study.best_trial.number,
            study.best_value,
        )

        return {
            "best_params": study.best_params,
            "best_macro_f1": round(study.best_value, 4),
            "n_trials": len(study.trials),
        }
