import numpy as np
import pandas as pd
import pytest

from financial_forecasting_platform.features.engineering import (
    create_date_features,
    create_distance_from_sma_feature,
    create_ema_crossover_feature,
    create_ema_feature,
    create_simple_moving_average_feature,
)


class TestCreateSimpleMovingAverageFeature:
    def test_adds_sma_column(self, sample_ohlcv_df):
        result = create_simple_moving_average_feature(sample_ohlcv_df, window_size=5)
        assert "SMA_5" in result.columns

    def test_sma_values(self, sample_ohlcv_df):
        window = 5
        result = create_simple_moving_average_feature(sample_ohlcv_df, window_size=window)
        for ticker in sample_ohlcv_df["Ticker"].unique():
            ticker_df = sample_ohlcv_df[sample_ohlcv_df["Ticker"] == ticker].sort_values("Date")
            sma = result[result["Ticker"] == ticker].sort_values("Date")[f"SMA_{window}"]
            expected = ticker_df["Close"].rolling(window=window, min_periods=window).mean()
            pd.testing.assert_series_equal(
                sma.dropna().reset_index(drop=True),
                expected.dropna().reset_index(drop=True),
                check_names=False,
                atol=1e-10,
            )

    def test_nan_for_insufficient_data(self, sample_ohlcv_df):
        window = 5
        result = create_simple_moving_average_feature(sample_ohlcv_df, window_size=window)
        for ticker in sample_ohlcv_df["Ticker"].unique():
            ticker_result = result[result["Ticker"] == ticker]
            assert ticker_result[f"SMA_{window}"].iloc[: window - 1].isna().all()

    def test_default_window(self, sample_ohlcv_df):
        result = create_simple_moving_average_feature(sample_ohlcv_df)
        assert "SMA_2" in result.columns

    def test_invalid_window_raises(self, sample_ohlcv_df):
        with pytest.raises(ValueError):
            create_simple_moving_average_feature(sample_ohlcv_df, window_size=0)

    def test_window_1_warns(self, sample_ohlcv_df):
        with pytest.warns(UserWarning):
            create_simple_moving_average_feature(sample_ohlcv_df, window_size=1)

    def test_does_not_modify_original(self, sample_ohlcv_df):
        original = sample_ohlcv_df.copy()
        create_simple_moving_average_feature(sample_ohlcv_df)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)


class TestCreateDistanceFromSmaFeature:
    def test_adds_distance_column(self, sample_ohlcv_df):
        result = create_distance_from_sma_feature(sample_ohlcv_df, window_size=5)
        assert "distance_close_vs_SMA_5" in result.columns

    def test_distance_values(self, sample_ohlcv_df):
        window = 5
        result = create_distance_from_sma_feature(sample_ohlcv_df, window_size=window)
        for ticker in sample_ohlcv_df["Ticker"].unique():
            ticker_df = sample_ohlcv_df[sample_ohlcv_df["Ticker"] == ticker].sort_values("Date")
            sma = ticker_df["Close"].rolling(window=window, min_periods=window).mean()
            expected = (ticker_df["Close"] - sma) / sma
            dist = result[result["Ticker"] == ticker].sort_values("Date")[f"distance_close_vs_SMA_{window}"]
            valid = expected.dropna().index.intersection(dist.dropna().index)
            pd.testing.assert_series_equal(
                dist.loc[valid].reset_index(drop=True),
                expected.loc[valid].reset_index(drop=True),
                check_names=False,
                atol=1e-10,
            )

    def test_invalid_window_raises(self, sample_ohlcv_df):
        with pytest.raises(ValueError):
            create_distance_from_sma_feature(sample_ohlcv_df, window_size=0)

    def test_does_not_modify_original(self, sample_ohlcv_df):
        original = sample_ohlcv_df.copy()
        create_distance_from_sma_feature(sample_ohlcv_df)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)


class TestCreateEmaFeature:
    def test_adds_ema_column(self, sample_ohlcv_df):
        result = create_ema_feature(sample_ohlcv_df, window_size=5)
        assert "ema_5" in result.columns

    def test_ema_values(self, sample_ohlcv_df):
        window = 5
        result = create_ema_feature(sample_ohlcv_df, window_size=window)
        for ticker in sample_ohlcv_df["Ticker"].unique():
            ticker_df = sample_ohlcv_df[sample_ohlcv_df["Ticker"] == ticker].sort_values("Date")
            expected = ticker_df["Close"].ewm(
                span=window, adjust=False, min_periods=window
            ).mean()
            ema = result[result["Ticker"] == ticker].sort_values("Date")[f"ema_{window}"]
            valid = expected.dropna().index.intersection(ema.dropna().index)
            pd.testing.assert_series_equal(
                ema.loc[valid].reset_index(drop=True),
                expected.loc[valid].reset_index(drop=True),
                check_names=False,
                atol=1e-10,
            )

    def test_invalid_window_raises(self, sample_ohlcv_df):
        with pytest.raises(ValueError):
            create_ema_feature(sample_ohlcv_df, window_size=0)

    def test_does_not_modify_original(self, sample_ohlcv_df):
        original = sample_ohlcv_df.copy()
        create_ema_feature(sample_ohlcv_df)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)


class TestCreateEmaCrossoverFeature:
    def test_adds_crossover_column(self, sample_ohlcv_df):
        result = create_ema_crossover_feature(sample_ohlcv_df, short_span=5, long_span=20)
        assert "ema5_minus_ema20" in result.columns

    def test_crossover_values(self, sample_ohlcv_df):
        short, long = 5, 20
        result = create_ema_crossover_feature(sample_ohlcv_df, short_span=short, long_span=long)
        for ticker in sample_ohlcv_df["Ticker"].unique():
            ticker_df = sample_ohlcv_df[sample_ohlcv_df["Ticker"] == ticker].sort_values("Date")
            ema_s = ticker_df["Close"].ewm(span=short, adjust=False, min_periods=short).mean()
            ema_l = ticker_df["Close"].ewm(span=long, adjust=False, min_periods=long).mean()
            expected = ema_s - ema_l
            col = result[result["Ticker"] == ticker].sort_values("Date")[f"ema{short}_minus_ema{long}"]
            valid = expected.dropna().index.intersection(col.dropna().index)
            pd.testing.assert_series_equal(
                col.loc[valid].reset_index(drop=True),
                expected.loc[valid].reset_index(drop=True),
                check_names=False,
                atol=1e-10,
            )

    def test_equal_spans_raises(self, sample_ohlcv_df):
        with pytest.raises(ValueError):
            create_ema_crossover_feature(sample_ohlcv_df, short_span=5, long_span=5)

    def test_negative_span_raises(self, sample_ohlcv_df):
        with pytest.raises(ValueError):
            create_ema_crossover_feature(sample_ohlcv_df, short_span=-1, long_span=10)

    def test_does_not_modify_original(self, sample_ohlcv_df):
        original = sample_ohlcv_df.copy()
        create_ema_crossover_feature(sample_ohlcv_df, short_span=5, long_span=20)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)


class TestCreateDateFeatures:
    def test_adds_day_of_week_and_month(self, sample_ohlcv_df):
        result = create_date_features(sample_ohlcv_df)
        assert "day_of_week" in result.columns
        assert "month" in result.columns

    def test_day_of_week_values(self, sample_ohlcv_df):
        result = create_date_features(sample_ohlcv_df)
        valid_days = {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday"}
        assert set(result["day_of_week"].unique()).issubset(valid_days)

    def test_month_values(self, sample_ohlcv_df):
        result = create_date_features(sample_ohlcv_df)
        months = result["month"].unique()
        for m in months:
            assert 1 <= int(m) <= 12

    def test_preserves_row_count(self, sample_ohlcv_df):
        result = create_date_features(sample_ohlcv_df)
        assert len(result) == len(sample_ohlcv_df)

    def test_does_not_modify_original(self, sample_ohlcv_df):
        original = sample_ohlcv_df.copy()
        create_date_features(sample_ohlcv_df)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)
