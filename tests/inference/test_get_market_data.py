# ruff: noqa: PLR2004
from unittest.mock import patch

import pandas as pd

from financial_forecasting_platform.inference.get_market_data import (
    _download_stock_fill,
    _ensure_stock_data,
)


def _db_window() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": pd.to_datetime(["2025-03-31", "2025-04-01"]),
            "Ticker": ["GOOG", "GOOG"],
            "Open": [100.0, 101.0],
            "High": [102.0, 103.0],
            "Low": [99.0, 99.5],
            "Close": [101.0, 102.0],
            "Volume": [1000000, 1100000],
        }
    )


def test_ensure_stock_data_tolerates_empty_fill_windows():
    """A fill window spanning only non-trading days returns empty from Yahoo and
    must not crash the prediction (regression for the 500 on /prediction)."""
    start = pd.Timestamp("2025-03-29")  # Saturday
    end = pd.Timestamp("2025-04-05")

    with (
        patch(
            "financial_forecasting_platform.inference.get_market_data."
            "get_stock_data_between_dates",
            return_value=_db_window(),
        ),
        patch(
            "financial_forecasting_platform.inference.get_market_data."
            "download_ohlcv_data",
            side_effect=ValueError("No data returned from Yahoo Finance"),
        ),
        patch(
            "financial_forecasting_platform.inference.get_market_data."
            "save_stock_data_to_database"
        ) as mock_save,
    ):
        result = _ensure_stock_data("GOOG", start, end)

    assert not result.empty
    mock_save.assert_not_called()


def test_ensure_stock_data_saves_backfilled_data():
    start = pd.Timestamp("2025-03-29")
    end = pd.Timestamp("2025-04-05")

    missing = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2025-03-31"]),
            "Ticker": ["GOOG"],
            "Open": [100.0],
            "High": [102.0],
            "Low": [99.0],
            "Close": [101.0],
            "Volume": [1000000],
        }
    )

    with (
        patch(
            "financial_forecasting_platform.inference.get_market_data."
            "get_stock_data_between_dates",
            return_value=_db_window(),
        ),
        patch(
            "financial_forecasting_platform.inference.get_market_data."
            "download_ohlcv_data",
            return_value=missing,
        ),
        patch(
            "financial_forecasting_platform.inference.get_market_data."
            "save_stock_data_to_database"
        ) as mock_save,
    ):
        _ensure_stock_data("GOOG", start, end)

    assert mock_save.call_count >= 1


def test_download_stock_fill_catches_value_error():
    with patch(
        "financial_forecasting_platform.inference.get_market_data."
        "download_ohlcv_data",
        side_effect=ValueError("No data returned from Yahoo Finance"),
    ):
        result = _download_stock_fill("GOOG", "2025-03-29", "2025-03-31")

    assert result.empty
