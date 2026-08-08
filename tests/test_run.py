"""
Tests for the Kedro project bootstrap and pipeline registry.
"""
from pathlib import Path

from kedro.framework.startup import bootstrap_project

from financial_forecasting_platform.pipeline_registry import register_pipelines

EXPECTED_PIPELINES = {
    "data_ingestion",
    "data_validation",
    "feature_engineering",
    "data_split",
    "outlier_handling",
    "model_training",
    "model_selection",
    "__default__",
}


def test_project_bootstraps():
    bootstrap_project(Path.cwd())


def test_pipeline_registry_registers_all_pipelines():
    bootstrap_project(Path.cwd())
    pipelines = register_pipelines()
    assert set(pipelines) == EXPECTED_PIPELINES
