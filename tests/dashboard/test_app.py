# ruff: noqa: PLR2004
import importlib
import pathlib
from unittest.mock import MagicMock, patch

import pytest
from streamlit.testing.v1 import AppTest

from financial_forecasting_platform.dashboard import app as dashboard_app
from financial_forecasting_platform.dashboard.api_client import ApiError

APP_PATH = str(pathlib.Path(dashboard_app.__file__))


def _rich_payload() -> dict:
    return {
        "ticker": "AAPL",
        "prediction": 1,
        "probability": 0.82,
        "confidence": "High",
        "close_price": 212.42,
        "model_type": "XGBoost",
        "model_version": "3",
        "model_stats": {
            "accuracy": 0.61,
            "precision": 0.64,
            "recall": 0.59,
            "roc_auc": 0.71,
        },
        "ohlcv": [
            {
                "date": "2026-08-01",
                "open": 210.0,
                "high": 213.0,
                "low": 209.5,
                "close": 212.0,
                "volume": 1000000,
            }
        ],
        "bollinger": {
            "dates": ["2026-08-01"],
            "upper": [220.0],
            "middle": [210.0],
            "lower": [200.0],
        },
        "features": {"rsi_14": 71.0, "vix_level": 14.9},
        "shap": [{"feature": "rsi_14", "contribution": -0.17}],
        "market": {"spy_return": 0.0082, "vix_level": 14.9, "regime": "Risk On"},
    }


def test_app_module_imports():
    """Regression: the app uses absolute imports so it runs as a streamlit script."""
    module = importlib.import_module(
        "financial_forecasting_platform.dashboard.app"
    )
    assert callable(module.main)


def test_app_initial_load_renders_without_exception():
    """Initial page load (no API call) renders the shell without errors."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()

    assert not at.exception
    assert [title.value for title in at.title] == [
        "Volatility Regime Prediction Dashboard"
    ]
    assert at.sidebar.selectbox
    assert at.sidebar.button
    assert at.info


def test_load_prediction_returns_payload():
    fake_client = MagicMock()
    fake_client.get_prediction.return_value = _rich_payload()

    with patch.object(
        dashboard_app, "_get_client", return_value=fake_client
    ):
        result = dashboard_app._load_prediction("AAPL")

    assert result["ticker"] == "AAPL"
    assert result["probability"] == 0.82
    fake_client.get_prediction.assert_called_once_with("AAPL")


def test_load_prediction_raises_api_error():
    fake_client = MagicMock()
    fake_client.get_prediction.side_effect = ApiError("API is down")

    with patch.object(
        dashboard_app, "_get_client", return_value=fake_client
    ):
        with pytest.raises(ApiError, match="API is down"):
            dashboard_app._load_prediction("AAPL")
