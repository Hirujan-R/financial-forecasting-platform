import numpy as np
import pandas as pd
import pytest

from financial_forecasting_platform.features.engineering import (
    create_close_lag_feature,
    create_lag_return_feature,
    create_log_return_lag_feature,
    create_market_movement_target,
    create_momentum_feature,
)


class TestCreateMarketMovementTarget:
    def test_returns_series_with_correct_name(self, sample_ohlcv_df):
        result = create_market_movement_target(sample_ohlcv_df)
        assert isinstance(result, pd.Series)
        assert result.name == "market_movement"

    def test_length_matches_input(self, sample_ohlcv_df):
        result = create_market_movement_target(sample_ohlcv_df)
        assert len(result) == len(sample_ohlcv_df)

    def test_values_are_binary_or_nan(self, sample_ohlcv_df):
        result = create_market_movement_target(sample_ohlcv_df)
        non_nan = result.dropna()
        assert set(non_nan.unique()).issubset({0.0, 1.0})

    def test_1_when_price_rises_next_day(self):
        dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
        df = pd.DataFrame({
            "Date": dates,
            "Ticker": "AAPL",
            "Open": [100.0, 102.0, 101.0],
            "High": [103.0, 104.0, 103.0],
            "Low": [99.0, 101.0, 100.0],
            "Close": [101.0, 103.0, 102.0],
            "Volume": [1_000_000, 1_000_000, 1_000_000],
        })
        result = create_market_movement_target(df)
        assert result.iloc[0] == 1.0
        assert result.iloc[1] == 0.0
        assert np.isnan(result.iloc[2])

    def test_0_when_price_drops_next_day(self):
        dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
        df = pd.DataFrame({
            "Date": dates,
            "Ticker": "AAPL",
            "Open": [100.0, 98.0, 99.0],
            "High": [101.0, 99.0, 100.0],
            "Low": [99.0, 97.0, 98.0],
            "Close": [100.0, 98.0, 99.0],
            "Volume": [1_000_000, 1_000_000, 1_000_000],
        })
        result = create_market_movement_target(df)
        assert result.iloc[0] == 0.0
        assert result.iloc[1] == 1.0
        assert np.isnan(result.iloc[2])

    def test_does_not_modify_original(self, sample_ohlcv_df):
        original = sample_ohlcv_df.copy()
        create_market_movement_target(sample_ohlcv_df)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)


class TestCreateCloseLagFeature:
    def test_adds_lag_column(self, sample_ohlcv_df):
        result = create_close_lag_feature(sample_ohlcv_df, lag=1)
        assert "close_lag_1" in result.columns

    def test_lag_values_are_shifted(self, sample_ohlcv_df):
        result = create_close_lag_feature(sample_ohlcv_df, lag=1)
        for ticker in sample_ohlcv_df["Ticker"].unique():
            ticker_df = sample_ohlcv_df[sample_ohlcv_df["Ticker"] == ticker].sort_values("Date")
            lag_col = result[result["Ticker"] == ticker].sort_values("Date")["close_lag_1"]
            pd.testing.assert_series_equal(
                lag_col.iloc[1:].reset_index(drop=True),
                ticker_df["Close"].iloc[:-1].reset_index(drop=True),
                check_names=False,
            )

    def test_first_row_per_ticker_is_nan(self, sample_ohlcv_df):
        result = create_close_lag_feature(sample_ohlcv_df, lag=1)
        for ticker in sample_ohlcv_df["Ticker"].unique():
            ticker_result = result[result["Ticker"] == ticker]
            assert np.isnan(ticker_result["close_lag_1"].iloc[0])

    def test_custom_lag(self, sample_ohlcv_df):
        result = create_close_lag_feature(sample_ohlcv_df, lag=5)
        assert "close_lag_5" in result.columns
        for ticker in sample_ohlcv_df["Ticker"].unique():
            ticker_result = result[result["Ticker"] == ticker]
            assert ticker_result["close_lag_5"].iloc[:5].isna().all()

    def test_invalid_lag_raises(self, sample_ohlcv_df):
        with pytest.raises(ValueError):
            create_close_lag_feature(sample_ohlcv_df, lag=0)

    def test_does_not_modify_original(self, sample_ohlcv_df):
        original = sample_ohlcv_df.copy()
        create_close_lag_feature(sample_ohlcv_df)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)


class TestCreateMomentumFeature:
    def test_adds_momentum_column(self, sample_ohlcv_df):
        result = create_momentum_feature(sample_ohlcv_df, lag=1)
        assert "momentum_1" in result.columns

    def test_momentum_values(self, sample_ohlcv_df):
        result = create_momentum_feature(sample_ohlcv_df, lag=1)
        for ticker in sample_ohlcv_df["Ticker"].unique():
            ticker_df = sample_ohlcv_df[sample_ohlcv_df["Ticker"] == ticker].sort_values("Date")
            diff = result[result["Ticker"] == ticker].sort_values("Date")["momentum_1"]
            expected = ticker_df["Close"].diff(1)
            pd.testing.assert_series_equal(
                diff.dropna().reset_index(drop=True),
                expected.dropna().reset_index(drop=True),
                check_names=False,
                atol=1e-10,
            )

    def test_custom_lag(self, sample_ohlcv_df):
        result = create_momentum_feature(sample_ohlcv_df, lag=3)
        assert "momentum_3" in result.columns

    def test_invalid_lag_raises(self, sample_ohlcv_df):
        with pytest.raises(ValueError):
            create_momentum_feature(sample_ohlcv_df, lag=0)

    def test_does_not_modify_original(self, sample_ohlcv_df):
        original = sample_ohlcv_df.copy()
        create_momentum_feature(sample_ohlcv_df)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)


class TestCreateLagReturnFeature:
    def test_adds_return_column(self, sample_ohlcv_df):
        result = create_lag_return_feature(sample_ohlcv_df, lag=1)
        assert "return_lag_1" in result.columns

    def test_return_values(self, sample_ohlcv_df):
        result = create_lag_return_feature(sample_ohlcv_df, lag=1)
        for ticker in sample_ohlcv_df["Ticker"].unique():
            ticker_df = sample_ohlcv_df[sample_ohlcv_df["Ticker"] == ticker].sort_values("Date")
            ret = result[result["Ticker"] == ticker].sort_values("Date")["return_lag_1"].dropna()
            prev_close = ticker_df["Close"].shift(1).dropna()
            expected = (ticker_df["Close"].reindex(prev_close.index) / prev_close) - 1
            pd.testing.assert_series_equal(
                ret.reset_index(drop=True),
                expected.reset_index(drop=True),
                check_names=False,
                atol=1e-10,
            )

    def test_invalid_lag_raises(self, sample_ohlcv_df):
        with pytest.raises(ValueError):
            create_lag_return_feature(sample_ohlcv_df, lag=0)

    def test_does_not_modify_original(self, sample_ohlcv_df):
        original = sample_ohlcv_df.copy()
        create_lag_return_feature(sample_ohlcv_df)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)


class TestCreateLogReturnLagFeature:
    def test_adds_log_return_column(self, sample_ohlcv_df):
        result = create_log_return_lag_feature(sample_ohlcv_df, lag=1)
        assert "log_return_lag_1" in result.columns

    def test_log_return_values(self, sample_ohlcv_df):
        result = create_log_return_lag_feature(sample_ohlcv_df, lag=1)
        for ticker in sample_ohlcv_df["Ticker"].unique():
            ticker_df = sample_ohlcv_df[sample_ohlcv_df["Ticker"] == ticker].sort_values("Date")
            log_ret = result[result["Ticker"] == ticker].sort_values("Date")["log_return_lag_1"].dropna()
            prev_close = ticker_df["Close"].shift(1).dropna()
            expected = np.log(
                ticker_df["Close"].reindex(prev_close.index) / prev_close
            )
            pd.testing.assert_series_equal(
                log_ret.reset_index(drop=True),
                expected.reset_index(drop=True),
                check_names=False,
                atol=1e-10,
            )

    def test_invalid_lag_raises(self, sample_ohlcv_df):
        with pytest.raises(ValueError):
            create_log_return_lag_feature(sample_ohlcv_df, lag=0)

    def test_does_not_modify_original(self, sample_ohlcv_df):
        original = sample_ohlcv_df.copy()
        create_log_return_lag_feature(sample_ohlcv_df)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)
