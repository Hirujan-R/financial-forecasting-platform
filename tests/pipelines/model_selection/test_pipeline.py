from unittest.mock import MagicMock, patch
import pytest
from kedro.pipeline import Pipeline

from financial_forecasting_platform.pipelines.model_selection.nodes import (
    assign_champion_alias,
    find_best_run,
    get_registered_model_from_run,
    get_runs,
)
from financial_forecasting_platform.pipelines.model_selection.pipeline import (
    create_pipeline,
)


@patch("financial_forecasting_platform.pipelines.model_selection.nodes.MlflowClient")
def test_get_runs(mock_mlflow_client):
    mock_client_instance = MagicMock()
    mock_mlflow_client.return_value = mock_client_instance

    mock_exp = MagicMock()
    mock_exp.experiment_id = "exp_123"
    mock_client_instance.get_experiment_by_name.return_value = mock_exp

    mock_runs = [MagicMock(), MagicMock()]
    mock_client_instance.search_runs.return_value = mock_runs

    runs = get_runs("test_exp")
    assert runs == mock_runs
    mock_client_instance.get_experiment_by_name.assert_called_once_with("test_exp")
    mock_client_instance.search_runs.assert_called_once_with(
        experiment_ids=["exp_123"],
        filter_string="tags.mlflow.parentRunId IS NOT NULL",
    )


def test_find_best_run():
    run1 = MagicMock()
    run1.info.run_id = "run_1"
    run1.data.metrics = {"roc_auc": 0.82}

    run2 = MagicMock()
    run2.info.run_id = "run_2"
    run2.data.metrics = {"roc_auc": 0.91}

    run3 = MagicMock()
    run3.info.run_id = "run_3"
    run3.data.metrics = {"roc_auc": 0.75}

    best_id = find_best_run([run1, run2, run3], metric="roc_auc")
    assert best_id == "run_2"


@patch("financial_forecasting_platform.pipelines.model_selection.nodes.MlflowClient")
def test_get_registered_model_from_run_success(mock_mlflow_client):
    mock_client_instance = MagicMock()
    mock_mlflow_client.return_value = mock_client_instance

    mock_version = MagicMock()
    mock_version.name = "XGBoostModel"
    mock_version.version = "2"
    mock_client_instance.search_model_versions.return_value = [mock_version]

    name, version = get_registered_model_from_run("run_2")
    assert name == "XGBoostModel"
    assert version == "2"
    mock_client_instance.search_model_versions.assert_called_once_with("run_id='run_2'")


@patch("financial_forecasting_platform.pipelines.model_selection.nodes.MlflowClient")
def test_get_registered_model_from_run_not_found(mock_mlflow_client):
    mock_client_instance = MagicMock()
    mock_mlflow_client.return_value = mock_client_instance
    mock_client_instance.search_model_versions.return_value = []

    with pytest.raises(Exception, match="No registered model found"):
        get_registered_model_from_run("run_nonexistent")


@patch("financial_forecasting_platform.pipelines.model_selection.nodes.MlflowClient")
def test_assign_champion_alias(mock_mlflow_client):
    mock_client_instance = MagicMock()
    mock_mlflow_client.return_value = mock_client_instance

    assign_champion_alias(model_version="3", model_name="MyModel")
    mock_client_instance.set_registered_model_alias.assert_called_once_with(
        name="MyModel",
        alias="champion",
        version="3",
    )


def test_create_pipeline():
    pipeline = create_pipeline()
    assert isinstance(pipeline, Pipeline)
    assert len(pipeline.nodes) == 4
    node_names = {n.name for n in pipeline.nodes}
    assert node_names == {
        "get_runs_node",
        "find_best_run_node",
        "get_registered_model_node",
        "assign_champion_node",
    }

