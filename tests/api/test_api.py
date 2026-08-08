# ruff: noqa: PLR2004
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from financial_forecasting_platform.api.main import app, get_predictor


def _slim_payload() -> dict:
    return {
        "ticker": "AAPL",
        "prediction": 1,
        "probability": 0.82,
        "confidence": "High",
        "close_price": 212.42,
        "model_type": "XGBoost",
        "model_version": "3",
    }


def _rich_payload() -> dict:
    payload = dict(_slim_payload())
    payload.update(
        {
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
            "features": {"rsi_14": 71.0, "Ticker": "AAPL", "vix_level": None},
            "shap": [
                {"feature": "rsi_14", "contribution": -0.17},
                {"feature": "volume", "contribution": 0.14},
            ],
            "market": {
                "spy_return": 0.0082,
                "vix_level": 18.2,
                "regime": "Risk On",
            },
        }
    )
    return payload


def _use_predictor_mock(payload) -> MagicMock:
    predictor = MagicMock()
    predictor.predict.return_value = payload
    predictor.get_prediction.return_value = payload
    app.dependency_overrides[get_predictor] = lambda: predictor
    return predictor


def _clear_overrides():
    app.dependency_overrides.clear()


def test_health_check():
    with TestClient(app) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@patch("financial_forecasting_platform.api.main.insert_prediction_log")
def test_predict_valid_ticker(mock_insert):
    predictor = _use_predictor_mock(_slim_payload())

    with TestClient(app) as client:
        response = client.post("/predict", json={"ticker": "AAPL"})

    _clear_overrides()

    assert response.status_code == 200
    assert response.json()["ticker"] == "AAPL"
    assert response.json()["prediction"] == 1
    assert response.json()["probability"] == 0.82

    predictor.predict.assert_called_once_with("AAPL")
    mock_insert.assert_called_once_with(
        ticker="AAPL",
        prediction=1,
        probability=0.82,
        model_name="XGBoost",
        model_version="3",
    )


def test_predict_invalid_ticker():
    _use_predictor_mock(_slim_payload())

    with TestClient(app) as client:
        response = client.post("/predict", json={"ticker": "TSLA"})

    _clear_overrides()

    assert response.status_code == 422


@patch("financial_forecasting_platform.api.main.insert_prediction_log")
def test_get_prediction_returns_rich_payload(mock_insert):
    predictor = _use_predictor_mock(_rich_payload())

    with TestClient(app) as client:
        response = client.get("/prediction/AAPL")

    _clear_overrides()

    assert response.status_code == 200
    body = response.json()
    assert body["ticker"] == "AAPL"
    assert body["model_stats"]["roc_auc"] == 0.71
    assert body["ohlcv"][0]["close"] == 212.0
    assert body["bollinger"]["upper"] == [220.0]
    assert body["features"]["rsi_14"] == 71.0
    assert body["shap"][0]["feature"] == "rsi_14"
    assert body["market"]["regime"] == "Risk On"

    predictor.get_prediction.assert_called_once_with("AAPL")
    mock_insert.assert_called_once_with(
        ticker="AAPL",
        prediction=1,
        probability=0.82,
        model_name="XGBoost",
        model_version="3",
    )


def test_get_prediction_invalid_ticker_returns_404():
    _use_predictor_mock(_rich_payload())

    with TestClient(app) as client:
        response = client.get("/prediction/TSLA")

    _clear_overrides()

    assert response.status_code == 404


@patch("financial_forecasting_platform.api.main.get_prediction_logs")
def test_history(mock_get_logs):
    mock_get_logs.return_value = [
        {
            "prediction_id": "uuid-1",
            "timestamp": "2026-08-06T12:00:00",
            "ticker": "AAPL",
            "prediction": 1,
            "probability": 0.82,
            "model_name": "XGBoost",
            "model_version": 3,
            "actual_outcome": None,
            "correct": None,
        }
    ]

    with TestClient(app) as client:
        response = client.get("/history", params={"limit": 10})

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["ticker"] == "AAPL"
    assert body[0]["probability"] == 0.82

    mock_get_logs.assert_called_once_with(10)
