"""
Feature Pipeline Re-exporter
=============================
Re-exports FeatureEngineer as FeaturePipeline for backwards compatibility.
"""

from src.features.feature_engineer import FeatureEngineer, FeaturePipeline

__all__ = ["FeatureEngineer", "FeaturePipeline"]
