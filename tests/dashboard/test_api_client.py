from unittest.mock import MagicMock, patch

import pytest
import requests

from financial_forecasting_platform.dashboard.api_client import ApiClient, ApiError


def _mock_response(status_code=200, payload=None):
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = payload
    response.text = "not found"
    return response


@patch("financial_forecasting_platform.dashboard.api_client.requests.get")
def test_get_prediction(mock_get):
    payload = {"ticker": "AAPL", "prediction": 1, "probability": 0.82}
    mock_get.return_value = _mock_response(200, payload)

    client = ApiClient(base_url="http://test", timeout=5)
    result = client.get_prediction("AAPL")

    assert result == payload
    mock_get.assert_called_once_with(
        "http://test/prediction/AAPL",
        params=None,
        timeout=5,
    )


@patch("financial_forecasting_platform.dashboard.api_client.requests.get")
def test_get_history_with_limit(mock_get):
    payload = [{"ticker": "AAPL", "probability": 0.82}]
    mock_get.return_value = _mock_response(200, payload)

    client = ApiClient(base_url="http://test", timeout=5)
    result = client.get_history(limit=25)

    assert result == payload
    mock_get.assert_called_once_with(
        "http://test/history",
        params={"limit": 25},
        timeout=5,
    )


@patch("financial_forecasting_platform.dashboard.api_client.requests.get")
def test_health_check(mock_get):
    mock_get.return_value = _mock_response(200, {"status": "ok"})

    client = ApiClient(base_url="http://test", timeout=5)
    assert client.health_check() == {"status": "ok"}


@patch("financial_forecasting_platform.dashboard.api_client.requests.get")
def test_http_error_raises_api_error(mock_get):
    mock_get.return_value = _mock_response(404, None)

    client = ApiClient(base_url="http://test", timeout=5)

    with pytest.raises(ApiError, match="status 404"):
        client.get_prediction("TSLA")


@patch("financial_forecasting_platform.dashboard.api_client.requests.get")
def test_network_error_raises_api_error(mock_get):
    mock_get.side_effect = requests.ConnectionError("refused")

    client = ApiClient(base_url="http://test", timeout=5)

    with pytest.raises(ApiError, match="Could not reach the prediction API"):
        client.get_prediction("AAPL")


def test_base_url_strips_trailing_slash():
    client = ApiClient(base_url="http://test/", timeout=5)
    assert client.base_url == "http://test"
