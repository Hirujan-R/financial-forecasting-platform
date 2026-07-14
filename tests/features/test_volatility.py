import numpy as np
import pandas as pd
import pytest

from financial_forecasting_platform.features.engineering import (
    create_bollinger_bands_features,
    create_log_return_volatility_feature,
    create_simple_return_volatility_feature,
)


class TestCreateSimpleReturnVolatilityFeature:
    def test_adds_volatility_column(self, sample_ohlcv_df):
        result = create_simple_return_volatility_feature(sample_ohlcv_df, window_size=5)
        assert "simple_return_volatility_5" in result.columns

    def test_volatility_values(self, sample_ohlcv_df):
        window = 5
        result = create_simple_return_volatility_feature(sample_ohlcv_df, window_size=window)
        for ticker in sample_ohlcv_df["Ticker"].unique():
            ticker_df = sample_ohlcv_df[sample_ohlcv_df["Ticker"] == ticker].sort_index()
            prev = ticker_df["Close"].shift(1)
            daily_ret = (ticker_df["Close"] - prev) / prev
            expected = daily_ret.rolling(window=window).std()
            vol = result[result["Ticker"] == ticker].sort_index()[f"simple_return_volatility_{window}"]
            valid = expected.dropna().index.intersection(vol.dropna().index)
            pd.testing.assert_series_equal(
                vol.loc[valid].reset_index(drop=True),
                expected.loc[valid].reset_index(drop=True),
                check_names=False,
                atol=1e-10,
            )

    def test_volatility_non_negative(self, sample_ohlcv_df):
        result = create_simple_return_volatility_feature(sample_ohlcv_df, window_size=5)
        assert (result["simple_return_volatility_5"].dropna() >= 0).all()

    def test_invalid_window_raises(self, sample_ohlcv_df):
        with pytest.raises(ValueError):
            create_simple_return_volatility_feature(sample_ohlcv_df, window_size=1)

    def test_does_not_modify_original(self, sample_ohlcv_df):
        original = sample_ohlcv_df.copy()
        create_simple_return_volatility_feature(sample_ohlcv_df)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)


class TestCreateLogReturnVolatilityFeature:
    def test_adds_volatility_column(self, sample_ohlcv_df):
        result = create_log_return_volatility_feature(sample_ohlcv_df, window_size=5)
        assert "log_return_volatility_5" in result.columns

    def test_volatility_values(self, sample_ohlcv_df):
        window = 5
        result = create_log_return_volatility_feature(sample_ohlcv_df, window_size=window)
        for ticker in sample_ohlcv_df["Ticker"].unique():
            ticker_df = sample_ohlcv_df[sample_ohlcv_df["Ticker"] == ticker].sort_index()
            prev = ticker_df["Close"].shift(1)
            log_ret = np.log(ticker_df["Close"] / prev)
            expected = log_ret.rolling(window=window).std()
            vol = result[result["Ticker"] == ticker].sort_index()[f"log_return_volatility_{window}"]
            valid = expected.dropna().index.intersection(vol.dropna().index)
            pd.testing.assert_series_equal(
                vol.loc[valid].reset_index(drop=True),
                expected.loc[valid].reset_index(drop=True),
                check_names=False,
                atol=1e-10,
            )

    def test_volatility_non_negative(self, sample_ohlcv_df):
        result = create_log_return_volatility_feature(sample_ohlcv_df, window_size=5)
        assert (result["log_return_volatility_5"].dropna() >= 0).all()

    def test_invalid_window_raises(self, sample_ohlcv_df):
        with pytest.raises(ValueError):
            create_log_return_volatility_feature(sample_ohlcv_df, window_size=1)

    def test_does_not_modify_original(self, sample_ohlcv_df):
        original = sample_ohlcv_df.copy()
        create_log_return_volatility_feature(sample_ohlcv_df)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)


class TestCreateBollingerBandsFeatures:
    def test_adds_three_columns(self, sample_ohlcv_df):
        result = create_bollinger_bands_features(sample_ohlcv_df, window_size=20)
        assert "bollinger_upper_distance_20" in result.columns
        assert "bollinger_lower_distance_20" in result.columns
        assert "bollinger_bandwidth_20" in result.columns

    def test_upper_distance_values(self, sample_ohlcv_df):
        window = 20
        result = create_bollinger_bands_features(sample_ohlcv_df, window_size=window)
        for ticker in sample_ohlcv_df["Ticker"].unique():
            ticker_df = sample_ohlcv_df[sample_ohlcv_df["Ticker"] == ticker].sort_index()
            sma = ticker_df["Close"].rolling(window=window, min_periods=window).mean()
            std = ticker_df["Close"].rolling(window=window, min_periods=window).std()
            upper = sma + 2 * std
            expected = (upper - ticker_df["Close"]) / ticker_df["Close"]
            col = result[result["Ticker"] == ticker].sort_index()[f"bollinger_upper_distance_{window}"]
            valid = expected.dropna().index.intersection(col.dropna().index)
            pd.testing.assert_series_equal(
                col.loc[valid].reset_index(drop=True),
                expected.loc[valid].reset_index(drop=True),
                check_names=False,
                atol=1e-10,
            )

    def test_lower_distance_values(self, sample_ohlcv_df):
        window = 20
        result = create_bollinger_bands_features(sample_ohlcv_df, window_size=window)
        for ticker in sample_ohlcv_df["Ticker"].unique():
            ticker_df = sample_ohlcv_df[sample_ohlcv_df["Ticker"] == ticker].sort_index()
            sma = ticker_df["Close"].rolling(window=window, min_periods=window).mean()
            std = ticker_df["Close"].rolling(window=window, min_periods=window).std()
            lower = sma - 2 * std
            expected = (ticker_df["Close"] - lower) / ticker_df["Close"]
            col = result[result["Ticker"] == ticker].sort_index()[f"bollinger_lower_distance_{window}"]
            valid = expected.dropna().index.intersection(col.dropna().index)
            pd.testing.assert_series_equal(
                col.loc[valid].reset_index(drop=True),
                expected.loc[valid].reset_index(drop=True),
                check_names=False,
                atol=1e-10,
            )

    def test_bandwidth_non_negative(self, sample_ohlcv_df):
        result = create_bollinger_bands_features(sample_ohlcv_df, window_size=20)
        assert (result["bollinger_bandwidth_20"].dropna() >= 0).all()

    def test_invalid_window_raises(self, sample_ohlcv_df):
        with pytest.raises(ValueError):
            create_bollinger_bands_features(sample_ohlcv_df, window_size=1)

    def test_does_not_modify_original(self, sample_ohlcv_df):
        original = sample_ohlcv_df.copy()
        create_bollinger_bands_features(sample_ohlcv_df)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)
