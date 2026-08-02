from unittest.mock import MagicMock, patch
import pandas as pd
from sklearn.pipeline import Pipeline

from financial_forecasting_platform.pipelines.model_training.nodes import (
    create_lr_pipeline,
    create_xgb_pipeline,
    train_model,
)


def test_create_lr_pipeline():
    X = pd.DataFrame(
        {
            "Ticker": ["AAPL", "AAPL"],
            "day_of_week": ["Monday", "Tuesday"],
            "log_return_volatility_5": [0.1, 0.2],
            "feat_other": [1.0, 2.0],
        }
    )
    pipe = create_lr_pipeline(X)
    assert isinstance(pipe, Pipeline)


def test_create_xgb_pipeline():
    pipe = create_xgb_pipeline()
    assert isinstance(pipe, Pipeline)


@patch("mlflow.evaluate")
@patch("mlflow.data.from_pandas")
@patch("mlflow.sklearn.log_model")
@patch("mlflow.log_params")
@patch("mlflow.set_tags")
@patch("mlflow.start_run")
def test_train_model(
    mock_start_run,
    mock_set_tags,
    mock_log_params,
    mock_log_model,
    mock_from_pandas,
    mock_evaluate,
):
    mock_model_info = MagicMock()
    mock_model_info.model_uri = "runs:/test/model"
    mock_log_model.return_value = mock_model_info

    X_train = pd.DataFrame(
        {
            "Ticker": ["AAPL"] * 10,
            "day_of_week": ["Monday"] * 10,
            "feat1": range(10),
        }
    )
    y_train = pd.Series([0, 1] * 5)
    X_test = X_train.copy()
    y_test = y_train.copy()

    pipeline = create_xgb_pipeline()
    param_grid = {"xgb__n_estimators": [1]}
    tags = {"model": "test_model"}

    uri = train_model(tags, X_train, y_train, X_test, y_test, pipeline, param_grid)
    assert uri == "runs:/test/model"

