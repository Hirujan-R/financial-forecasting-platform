# ruff: noqa: PLR2004
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from financial_forecasting_platform.inference.predictor import (
    VolatilityPredictor,
)


def _make_stock_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": pd.date_range("2026-01-01", periods=20),
            "Ticker": ["AAPL"] * 20,
            "Open": np.linspace(100, 120, 20),
            "High": np.linspace(101, 122, 20),
            "Low": np.linspace(99, 118, 20),
            "Close": np.linspace(100.5, 121, 20),
            "Volume": [1_000_000] * 20,
        }
    )


def _make_engineered_data() -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=200, name="Date")
    data = {
        "Ticker": ["AAPL"] * 200,
        "day_of_week": pd.Categorical(["Monday"] * 200),
        "spy_lag_return_1": np.random.default_rng(42).normal(size=200),
        "vix_level": np.linspace(15, 25, 200),
    }
    for i in range(1, 21):
        data[f"feature_{i}"] = np.random.default_rng(i).normal(size=200)
    return pd.DataFrame(data, index=index)


def _build_predictor(model_type: str = "XGBoost"):
    with (
        patch("financial_forecasting_platform.inference.predictor.load_champion_model") as mock_load,
        patch("financial_forecasting_platform.inference.predictor.MlflowClient") as mock_client_cls,
    ):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        version = MagicMock()
        version.tags = {"model": model_type}
        version.version = "3"
        version.run_id = "run_123"
        mock_client.get_model_version_by_alias.return_value = version

        run = MagicMock()
        run.data.metrics = {
            "accuracy": 0.61,
            "precision": 0.64,
            "recall": 0.59,
            "roc_auc": 0.71,
        }
        mock_client.get_run.return_value = run

        model = MagicMock()
        mock_load.return_value = model

        predictor = VolatilityPredictor()

    return predictor, model


@pytest.mark.parametrize(
    "probability,expected",
    [
        (0.95, "High"),
        (0.75, "High"),
        (0.7, "Medium"),
        (0.6, "Medium"),
        (0.4, "Low"),
    ],
)
def test_confidence_thresholds(probability, expected):
    assert VolatilityPredictor._confidence(probability) == expected


def test_init_loads_model_stats():
    predictor, _ = _build_predictor()

    assert predictor.model_stats == {
        "accuracy": 0.61,
        "precision": 0.64,
        "recall": 0.59,
        "roc_auc": 0.71,
    }


def test_predict_returns_slim_payload():
    predictor, model = _build_predictor()
    stock_data = _make_stock_data()
    engineered = _make_engineered_data()

    model.predict.return_value = [1]
    model.predict_proba.return_value = np.array([[0.18, 0.82]])

    with patch.object(predictor, "_prepare_features", return_value=(stock_data, engineered)):
        result = predictor.predict("AAPL")

    assert result["ticker"] == "AAPL"
    assert result["prediction"] == 1
    assert result["probability"] == 0.82
    assert result["confidence"] == "High"
    assert result["close_price"] == pytest.approx(121.0)
    assert result["model_type"] == "XGBoost"
    assert result["model_version"] == "3"


def test_get_prediction_composes_rich_payload():
    predictor, model = _build_predictor()
    stock_data = _make_stock_data()
    engineered = _make_engineered_data()

    model.predict.return_value = [0]
    model.predict_proba.return_value = np.array([[0.72, 0.28]])

    fixed_shap = [
        {"feature": "feature_5", "contribution": 0.42},
        {"feature": "feature_1", "contribution": -0.17},
    ]

    with (
        patch.object(predictor, "_prepare_features", return_value=(stock_data, engineered)),
        patch.object(predictor, "_explain_features", return_value=fixed_shap),
    ):
        result = predictor.get_prediction("AAPL", top_n=2)

    assert result["prediction"] == 0
    assert result["probability"] == 0.28
    assert result["shap"] == fixed_shap
    assert result["model_stats"] == predictor.model_stats
    assert result["market"]["regime"] in {"Risk On", "Elevated", "Risk Off"}
    assert result["market"]["vix_level"] is not None
    assert result["market"]["spy_return"] is not None
    assert len(result["ohlcv"]) == 20
    assert result["ohlcv"][0]["open"] == pytest.approx(100.0)
    assert "upper" in result["bollinger"]
    assert result["bollinger"]["middle"][-1] is not None
    assert "feature_1" in result["features"]


def test_explain_delegates_to_explain_features():
    predictor, _ = _build_predictor()
    engineered = _make_engineered_data()

    fixed_shap = [{"feature": "feature_2", "contribution": 0.31}]

    with (
        patch.object(predictor, "_prepare_features", return_value=(MagicMock(), engineered)),
        patch.object(predictor, "_explain_features", return_value=fixed_shap),
    ):
        result = predictor.explain("AAPL", top_n=1)

    assert result == fixed_shap


def test_explain_features_xgboost():
    predictor, model = _build_predictor(model_type="XGBoost")
    engineered = _make_engineered_data()

    preprocessor = MagicMock()
    preprocessor.transform.side_effect = [
        np.zeros((150, 5)),
        np.zeros((1, 5)),
    ]
    preprocessor.get_feature_names_out.return_value = np.array(
        ["f1", "f2", "f3", "f4", "f5"]
    )

    model.named_steps = {
        "preprocessor": preprocessor,
        "xgb": MagicMock(),
    }

    mock_explainer = MagicMock()
    mock_explainer.shap_values.return_value = np.array([[0.5, -0.2, 0.1, 0.0, -0.4]])

    with patch(
        "financial_forecasting_platform.inference.predictor.shap.TreeExplainer",
        return_value=mock_explainer,
    ):
        importance = predictor._explain_features(engineered, top_n=3)

    assert len(importance) == 3
    assert importance[0]["feature"] == "f1"
    assert importance[0]["contribution"] == pytest.approx(0.5)
    assert {item["feature"] for item in importance} == {"f1", "f5", "f2"}


def test_explain_features_coerces_object_dtype_to_float():
    predictor, model = _build_predictor(model_type="XGBoost")
    engineered = _make_engineered_data()

    preprocessor = MagicMock()
    preprocessor.transform.side_effect = [
        np.array([["01", "02", "03", 1.5, 2.0]] * 150, dtype=object),
        np.array([["04", "05", "06", 3.0, 4.0]], dtype=object),
    ]
    preprocessor.get_feature_names_out.return_value = np.array(
        ["f1", "f2", "f3", "f4", "f5"]
    )

    model.named_steps = {
        "preprocessor": preprocessor,
        "xgb": MagicMock(),
    }

    mock_explainer = MagicMock()
    mock_explainer.shap_values.return_value = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]])

    with patch(
        "financial_forecasting_platform.inference.predictor.shap.TreeExplainer",
        return_value=mock_explainer,
    ) as mock_tree:
        predictor._explain_features(engineered, top_n=2)

    call_kwargs = mock_tree.call_args.kwargs
    assert call_kwargs["data"].dtype == float
    assert call_kwargs["data"].shape == (150, 5)


def test_explain_features_logistic_regression():
    predictor, model = _build_predictor(model_type="Logistic Regression")
    engineered = _make_engineered_data()

    preprocessor = MagicMock()
    preprocessor.transform.side_effect = [
        np.zeros((150, 4)),
        np.zeros((1, 4)),
    ]
    preprocessor.get_feature_names_out.return_value = np.array(
        ["f1", "f2", "f3", "f4"]
    )

    model.named_steps = {
        "preprocessor": preprocessor,
        "logreg": MagicMock(),
    }

    mock_explainer = MagicMock()
    mock_explainer.shap_values.return_value = np.array([[0.1, -0.3, 0.2, 0.0]])

    with patch(
        "financial_forecasting_platform.inference.predictor.shap.LinearExplainer",
        return_value=mock_explainer,
    ):
        importance = predictor._explain_features(engineered, top_n=2)

    assert len(importance) == 2
    assert importance[0]["feature"] == "f2"
    assert importance[0]["contribution"] == pytest.approx(-0.3)
