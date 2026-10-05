"""
Model Registry Loader
=====================
Loads champion and challenger models from MLflow registry at startup,
and provides thread-safe access for concurrent requests.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

try:
    import mlflow
    import mlflow.sklearn
    HAS_MLFLOW = True
except ImportError:
    mlflow = None
    HAS_MLFLOW = False

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Thread-safe model registry for champion/challenger serving."""

    def __init__(self):
        self._lock = threading.RLock()
        self._champion: Any = None
        self._challenger: Any = None
        self._champion_version: str = "unknown"
        self._challenger_version: str = "unknown"
        self._champion_name: str = ""
        self._challenger_name: str = ""

    def load_champion(self, model_name: str, stage: str = "Production") -> None:
        """Load the production champion model from MLflow registry."""
        with self._lock:
            try:
                model_uri = f"models:/{model_name}/{stage}"
                model = mlflow.sklearn.load_model(model_uri)
                self._champion = model
                self._champion_name = model_name

                # Get version info
                client = mlflow.tracking.MlflowClient()
                versions = client.get_latest_versions(model_name, stages=[stage])
                self._champion_version = versions[0].version if versions else "1"
                logger.info(
                    f"Champion loaded: {model_name} v{self._champion_version}"
                )
            except Exception as e:
                logger.warning(f"Could not load champion from registry: {e}")
                # Fallback: try latest version
                try:
                    model_uri = f"models:/{model_name}/latest"
                    model = mlflow.sklearn.load_model(model_uri)
                    self._champion = model
                    self._champion_name = model_name
                    self._champion_version = "latest"
                    logger.info(f"Champion loaded (latest): {model_name}")
                except Exception as e2:
                    logger.warning(f"Could not load any champion: {e2}")
                    raise

    def load_challenger(self, model_name: str, stage: str = "Staging") -> None:
        """Load the challenger model for shadow mode."""
        with self._lock:
            try:
                model_uri = f"models:/{model_name}/{stage}"
                model = mlflow.sklearn.load_model(model_uri)
                self._challenger = model
                self._challenger_name = model_name
                logger.info(f"Challenger loaded: {model_name}")
            except Exception as e:
                logger.debug(f"No challenger model: {e}")

    def predict(self, X) -> Any:
        """Get champion model prediction."""
        with self._lock:
            if self._champion is None:
                raise RuntimeError("No champion model loaded")
            return self._champion.predict_proba(X)

    def predict_challenger(self, X) -> Any | None:
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
    def champion_version(self) -> str:
        return self._champion_version

    @property
    def champion_name(self) -> str:
        return self._champion_name

    @property
    def challenger_loaded(self) -> bool:
        return self._challenger is not None

    @property
    def challenger_name(self) -> str:
        return self._challenger_name

    def reload_champion(self, model_name: str) -> None:
        """Hot-reload the champion model (called after promotion)."""
        logger.info(f"Hot-reloading champion: {model_name}")
        self.load_champion(model_name)
