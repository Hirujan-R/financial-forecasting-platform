"""Project pipelines."""
from __future__ import annotations

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
import financial_forecasting_platform.pipelines.data_ingestion.pipeline as data_ingestion
import financial_forecasting_platform.pipelines.feature_engineering.pipeline as feature_engineering



def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    data_ingestion_pipeline = data_ingestion.create_pipeline()
    feature_engineering_pipeline = feature_engineering.create_pipeline()

    pipelines = {}
    pipelines["data_ingestion"] = data_ingestion_pipeline
    pipelines["feature_engineering"] = feature_engineering_pipeline
    pipelines["__default__"] = sum(pipelines.values())
    return pipelines
