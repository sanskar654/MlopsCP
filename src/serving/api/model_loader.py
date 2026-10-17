"""
Model Registry Loader (Modern MLflow Aliases Architecture)
===========================================================
Loads champion and challenger models from MLflow model registry using
modern MLflow Aliases (`@champion`, `@challenger`) rather than deprecated stages.
Provides thread-safe access for concurrent FastAPI worker requests.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Optional

try:
    import mlflow
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient
    HAS_MLFLOW = True
except ImportError:
    mlflow = None
    HAS_MLFLOW = False

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Thread-safe model registry loader using MLflow aliases."""

    def __init__(self):
        self._lock = threading.RLock()
        self._champion: Any = None
        self._challenger: Any = None
        self._champion_version: str = "unknown"
        self._challenger_version: str = "unknown"
        self._champion_name: str = ""
        self._challenger_name: str = ""

    def load_champion(self, model_name: str, alias: str = "champion") -> None:
        """
        Load the champion model using MLflow model alias (e.g., @champion).
        """
        with self._lock:
            if not HAS_MLFLOW:
                logger.warning("MLflow not installed; skipping remote registry load.")
                return

            try:
                # 1. Primary: load using modern MLflow alias URI
                model_uri = f"models:/{model_name}@{alias}"
                logger.info(f"Loading champion model via alias URI: {model_uri}")
                model = mlflow.sklearn.load_model(model_uri)
                self._champion = model
                self._champion_name = model_name

                # Retrieve version for alias
                client = MlflowClient()
                mv = client.get_model_version_by_alias(model_name, alias)
                self._champion_version = mv.version if mv else "champion"
                logger.info(f"Champion loaded: {model_name} (version {self._champion_version}, alias @{alias})")
            except Exception as e_alias:
                logger.warning(f"Could not load champion via @{alias} ({e_alias}). Trying latest fallback...")
                try:
                    # 2. Fallback: load latest registered version
                    client = MlflowClient()
                    latest_versions = client.get_latest_versions(model_name)
                    if latest_versions:
                        latest_v = latest_versions[0].version
                        model_uri = f"models:/{model_name}/{latest_v}"
                        self._champion = mlflow.sklearn.load_model(model_uri)
                        self._champion_name = model_name
                        self._champion_version = latest_v
                        # Assign champion alias automatically
                        try:
                            client.set_registered_model_alias(model_name, alias, latest_v)
                            logger.info(f"Assigned alias @{alias} to version {latest_v}")
                        except Exception:
                            pass
                        logger.info(f"Champion loaded (fallback version {latest_v}): {model_name}")
                    else:
                        raise RuntimeError(f"No versions found for model {model_name}")
                except Exception as e_fallback:
                    logger.warning(f"Could not load champion model: {e_fallback}. Running in fallback heuristic mode.")

    def load_challenger(self, model_name: str, alias: str = "challenger") -> None:
        """
        Load the challenger model using MLflow model alias (@challenger) for shadow mode.
        """
        with self._lock:
            if not HAS_MLFLOW:
                return

            try:
                model_uri = f"models:/{model_name}@{alias}"
                self._challenger = mlflow.sklearn.load_model(model_uri)
                self._challenger_name = model_name

                client = MlflowClient()
                mv = client.get_model_version_by_alias(model_name, alias)
                self._challenger_version = mv.version if mv else "challenger"
                logger.info(f"Challenger loaded: {model_name} (version {self._challenger_version}, alias @{alias})")
            except Exception as e:
                logger.debug(f"Challenger model not loaded: {e}")

    def predict(self, X) -> Any:
        """Get champion model prediction."""
        with self._lock:
            if self._champion is None:
                raise RuntimeError("No champion model loaded")
            return self._champion.predict_proba(X)

    def predict_challenger(self, X) -> Optional[Any]:
        """Get challenger model prediction (shadow mode)."""
        with self._lock:
            if self._challenger is None:
                return None
            try:
                return self._challenger.predict_proba(X)
            except Exception as e:
                logger.debug(f"Challenger prediction failed: {e}")
                return None

    @property
    def champion_loaded(self) -> bool:
        return self._champion is not None

    @property
    def challenger_loaded(self) -> bool:
        return self._challenger is not None

    @property
    def champion_version(self) -> str:
        return self._champion_version

    @property
    def challenger_version(self) -> str:
        return self._challenger_version

    @property
    def champion_name(self) -> str:
        return self._champion_name

    @property
    def challenger_name(self) -> str:
        return self._challenger_name
