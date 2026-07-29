import numpy as np
import pandas as pd
import pytest
from kedro.pipeline import Pipeline
from unittest.mock import patch, MagicMock

from financial_forecasting_platform.pipelines.data_ingestion.nodes import (
    download_ohlcv_data,
    download_market_data,
    create_spy_data,
    create_vix_data,
)
from financial_forecasting_platform.pipelines.data_ingestion.pipeline import (
    create_pipeline,
)


def _make_yf_response(tickers: list[str], dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Build a DataFrame mimicking yfinance download output with group_by='Ticker'.
    yfinance always returns MultiIndex columns (Ticker, Price) when group_by='Ticker'.
    The index is named 'Date' and column level 0 is named 'Ticker'."""
    price_cols = ["Close", "Open", "High", "Low", "Volume"]
    arrays = []
    for ticker in tickers:
        for col in price_cols:
            arrays.append((ticker, col))
    columns = pd.MultiIndex.from_tuples(arrays, names=["Ticker", None])
    named_dates = dates.rename("Date")
    n_cols = len(tickers) * len(price_cols)
    data = np.arange(len(dates) * n_cols, dtype=float).reshape(len(dates), n_cols)
    return pd.DataFrame(data, index=named_dates, columns=columns)


class TestDownloadOhlcvData:
    @patch("financial_forecasting_platform.pipelines.data_ingestion.nodes.yf.download")
    def test_returns_dataframe_with_expected_columns(self, mock_download):
        dates = pd.date_range("2024-01-01", periods=3, freq="B")
        mock_download.return_value = _make_yf_response(["AAPL"], dates)

        result = download_ohlcv_data(["AAPL"], "2024-01-01", "2024-01-05")

        assert isinstance(result, pd.DataFrame)
        assert "Date" in result.columns
        assert "Ticker" in result.columns
        assert set(["Open", "High", "Low", "Close", "Volume"]).issubset(result.columns)

    @patch("financial_forecasting_platform.pipelines.data_ingestion.nodes.yf.download")
    def test_date_and_ticker_are_columns_not_index(self, mock_download):
        dates = pd.date_range("2024-01-01", periods=3, freq="B")
        mock_download.return_value = _make_yf_response(["AAPL"], dates)

        result = download_ohlcv_data(["AAPL"], "2024-01-01", "2024-01-05")

        assert result.index.name is None or "Date" not in result.index.names

    @patch("financial_forecasting_platform.pipelines.data_ingestion.nodes.yf.download")
    def test_row_count_matches_dates(self, mock_download):
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        mock_download.return_value = _make_yf_response(["AAPL"], dates)

        result = download_ohlcv_data(["AAPL"], "2024-01-01", "2024-01-10")

        assert len(result) == 5

    @patch("financial_forecasting_platform.pipelines.data_ingestion.nodes.yf.download")
    def test_multiple_tickers(self, mock_download):
        dates = pd.date_range("2024-01-01", periods=3, freq="B")
        mock_download.return_value = _make_yf_response(["AAPL", "MSFT"], dates)

        result = download_ohlcv_data(["AAPL", "MSFT"], "2024-01-01", "2024-01-05")

        assert set(result["Ticker"].unique()) == {"AAPL", "MSFT"}
        assert len(result) == 6

    @patch("financial_forecasting_platform.pipelines.data_ingestion.nodes.yf.download")
    def test_tickers_are_uppercased(self, mock_download):
        dates = pd.date_range("2024-01-01", periods=2, freq="B")
        mock_download.return_value = _make_yf_response(["AAPL"], dates)

        download_ohlcv_data(["aapl"], "2024-01-01", "2024-01-03")

        call_args = mock_download.call_args
        assert call_args[0][0] == ["AAPL"]

    def test_empty_tickers_raises(self):
        with pytest.raises(ValueError, match="tickers cannot be empty"):
            download_ohlcv_data([], "2024-01-01", "2024-01-05")

    def test_non_string_tickers_raises(self):
        with pytest.raises(TypeError, match="tickers must contain strings"):
            download_ohlcv_data([123, 456], "2024-01-01", "2024-01-05")

    def test_start_after_end_raises(self):
        with pytest.raises(ValueError, match="start_date must be before end_date"):
            download_ohlcv_data(["AAPL"], "2024-01-10", "2024-01-01")

    def test_start_equals_end_raises(self):
        with pytest.raises(ValueError, match="start_date must be before end_date"):
            download_ohlcv_data(["AAPL"], "2024-01-01", "2024-01-01")

    @patch("financial_forecasting_platform.pipelines.data_ingestion.nodes.yf.download")
    def test_empty_response_raises(self, mock_download):
        mock_download.return_value = pd.DataFrame()

        with pytest.raises(ValueError, match="No data returned"):
            download_ohlcv_data(["AAPL"], "2024-01-01", "2024-01-05")

    @patch("financial_forecasting_platform.pipelines.data_ingestion.nodes.yf.download")
    def test_passes_correct_args_to_yfinance(self, mock_download):
        dates = pd.date_range("2024-01-01", periods=2, freq="B")
        mock_download.return_value = _make_yf_response(["AAPL"], dates)

        download_ohlcv_data(["AAPL"], "2024-01-01", "2024-01-05")

        mock_download.assert_called_once_with(
            ["AAPL"],
            group_by="Ticker",
            start="2024-01-01",
            end="2024-01-05",
            repair=True,
        )

    @patch("financial_forecasting_platform.pipelines.data_ingestion.nodes.yf.download")
    def test_does_not_modify_input_list(self, mock_download):
        dates = pd.date_range("2024-01-01", periods=2, freq="B")
        mock_download.return_value = _make_yf_response(["AAPL"], dates)
        tickers = ["aapl"]

        download_ohlcv_data(tickers, "2024-01-01", "2024-01-03")

        assert tickers == ["aapl"]


class TestDownloadMarketData:
    @patch("financial_forecasting_platform.pipelines.data_ingestion.nodes.yf.download")
    def test_default_tickers(self, mock_download):
        dates = pd.date_range("2024-01-01", periods=2, freq="B")
        mock_download.return_value = _make_yf_response(["SPY", "^VIX"], dates)

        result = download_market_data("2024-01-01", "2024-01-05")

        assert isinstance(result, pd.DataFrame)
        mock_download.assert_called_once_with(
            ["SPY", "^VIX"],
            start="2024-01-01",
            end="2024-01-05"
        )
        assert list(result.columns) == ["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]
        assert set(result["Ticker"].unique()) == {"SPY", "^VIX"}

    @patch("financial_forecasting_platform.pipelines.data_ingestion.nodes.yf.download")
    def test_custom_tickers(self, mock_download):
        dates = pd.date_range("2024-01-01", periods=2, freq="B")
        mock_download.return_value = _make_yf_response(["QQQ"], dates)

        result = download_market_data("2024-01-01", "2024-01-05", tickers=["QQQ"])

        assert isinstance(result, pd.DataFrame)
        mock_download.assert_called_once_with(
            ["QQQ"],
            start="2024-01-01",
            end="2024-01-05"
        )


class TestCreateSpyData:
    def test_filters_spy_ticker(self):
        df = pd.DataFrame(
            {
                "Date": ["2024-01-01", "2024-01-01"],
                "Ticker": ["SPY", "^VIX"],
                "Close": [500.0, 15.0],
            }
        )
        spy_df = create_spy_data(df)
        assert len(spy_df) == 1
        assert (spy_df["Ticker"] == "SPY").all()


class TestCreateVixData:
    def test_filters_vix_ticker(self):
        df = pd.DataFrame(
            {
                "Date": ["2024-01-01", "2024-01-01"],
                "Ticker": ["SPY", "^VIX"],
                "Close": [500.0, 15.0],
            }
        )
        vix_df = create_vix_data(df)
        assert len(vix_df) == 1
        assert (vix_df["Ticker"] == "^VIX").all()


class TestDataIngestionPipeline:
    def test_returns_pipeline(self):
        result = create_pipeline()
        assert isinstance(result, Pipeline)

    def test_pipeline_node_name(self):
        pipeline = create_pipeline()
        node_names = [n.name for n in pipeline.nodes]
        assert "download_stock_data_node" in node_names
        assert "download_market_data_node" in node_names
        assert "create_spy_data_node" in node_names
        assert "create_vix_data_node" in node_names

    def test_pipeline_outputs_raw_data(self):
        pipeline = create_pipeline()
        output_datasets = set()
        for n in pipeline.nodes:
            output_datasets.update(n.outputs)
        assert "raw_data" in output_datasets

    def test_pipeline_inputs_params(self):
        pipeline = create_pipeline()
        input_datasets = set()
        for n in pipeline.nodes:
            input_datasets.update(n.inputs)
        assert "params:tickers" in input_datasets
        assert "params:raw_data_start_date" in input_datasets
        assert "params:raw_data_end_date" in input_datasets

    def test_pipeline_has_four_nodes(self):
        pipeline = create_pipeline()
        assert len(pipeline.nodes) == 4
