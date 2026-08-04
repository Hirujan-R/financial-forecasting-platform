"""Project pipelines."""
from __future__ import annotations

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
import financial_forecasting_platform.pipelines.data_ingestion.pipeline as data_ingestion
import financial_forecasting_platform.pipelines.feature_engineering.pipeline as feature_engineering
import financial_forecasting_platform.pipelines.data_validation.pipeline as data_validation
import financial_forecasting_platform.pipelines.data_split as data_split
import financial_forecasting_platform.pipelines.outlier_handling as outlier_handling
import financial_forecasting_platform.pipelines.model_training.pipeline as model_training
import financial_forecasting_platform.pipelines.model_selection as model_selection

def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    data_ingestion_pipeline = data_ingestion.create_pipeline()
    data_validation_pipeline = data_validation.create_pipeline()
    feature_engineering_pipeline = feature_engineering.create_pipeline()
    data_split_pipeline = data_split.create_pipeline()
    outlier_handling_pipeline  = outlier_handling.create_pipeline()
    model_training_pipeline = model_training.create_pipeline()
    model_selection_pipeline = model_selection.create_pipeline()

    pipelines = {}
    pipelines["data_ingestion"] = data_ingestion_pipeline
    pipelines["data_validation"] = data_validation_pipeline
    pipelines["feature_engineering"] = feature_engineering_pipeline
    pipelines["data_split"] = data_split_pipeline
    pipelines["outlier_handling"] = outlier_handling_pipeline
    pipelines["model_training"] = model_training_pipeline
    pipelines["model_selection"] = model_selection_pipeline
    pipelines["__default__"] = sum(pipelines.values())
    
    return pipelines
