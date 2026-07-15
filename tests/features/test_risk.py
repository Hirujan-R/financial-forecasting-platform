import numpy as np
import pandas as pd
import pytest

from financial_forecasting_platform.features.engineering import (
    create_drawdown_feature,
    create_rolling_sharpe_ratio_feature,
    create_rolling_window_mdd_feature,
)


class TestCreateDrawdownFeature:
    def test_adds_drawdown_column(self, sample_ohlcv_df):
        result = create_drawdown_feature(sample_ohlcv_df, window_size=5)
        assert "drawdown_5" in result.columns

    def test_drawdown_values(self, sample_ohlcv_df):
        window = 5
        result = create_drawdown_feature(sample_ohlcv_df, window_size=window)
        for ticker in sample_ohlcv_df["Ticker"].unique():
            ticker_df = sample_ohlcv_df[sample_ohlcv_df["Ticker"] == ticker].sort_values("Date")
            peak = ticker_df["Close"].rolling(window=window, min_periods=window).max()
            expected = (ticker_df["Close"] - peak) / peak
            dd = result[result["Ticker"] == ticker].sort_values("Date")[f"drawdown_{window}"]
            valid = expected.dropna().index.intersection(dd.dropna().index)
            pd.testing.assert_series_equal(
                dd.loc[valid].reset_index(drop=True),
                expected.loc[valid].reset_index(drop=True),
                check_names=False,
                atol=1e-10,
            )

    def test_drawdown_non_positive(self, sample_ohlcv_df):
        result = create_drawdown_feature(sample_ohlcv_df, window_size=5)
        assert (result["drawdown_5"].dropna() <= 0).all()

    def test_default_drawdown_always_leq_zero(self, single_ticker_df):
        result = create_drawdown_feature(single_ticker_df, window_size=5)
        dd = result["drawdown_5"].dropna()
        assert (dd <= 0).all()

    def test_invalid_window_raises(self, sample_ohlcv_df):
        with pytest.raises(ValueError):
            create_drawdown_feature(sample_ohlcv_df, window_size=1)

    def test_does_not_modify_original(self, sample_ohlcv_df):
        original = sample_ohlcv_df.copy()
        create_drawdown_feature(sample_ohlcv_df)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)


class TestCreateRollingWindowMddFeature:
    def test_adds_mdd_column(self, sample_ohlcv_df):
        result = create_rolling_window_mdd_feature(sample_ohlcv_df, window_size=5)
        assert "rolling_window_mdd_5" in result.columns

    def test_mdd_values(self, sample_ohlcv_df):
        window = 5
        result = create_rolling_window_mdd_feature(sample_ohlcv_df, window_size=window)
        for ticker in sample_ohlcv_df["Ticker"].unique():
            ticker_df = sample_ohlcv_df[sample_ohlcv_df["Ticker"] == ticker].sort_values("Date")
            peak = ticker_df["Close"].rolling(window=window, min_periods=window).max()
            drawdown = (ticker_df["Close"] - peak) / peak
            expected = drawdown.rolling(window=window, min_periods=window).min()
            mdd = result[result["Ticker"] == ticker].sort_values("Date")[f"rolling_window_mdd_{window}"]
            valid = expected.dropna().index.intersection(mdd.dropna().index)
            pd.testing.assert_series_equal(
                mdd.loc[valid].reset_index(drop=True),
                expected.loc[valid].reset_index(drop=True),
                check_names=False,
                atol=1e-10,
            )

    def test_mdd_non_positive(self, sample_ohlcv_df):
        result = create_rolling_window_mdd_feature(sample_ohlcv_df, window_size=5)
        assert (result["rolling_window_mdd_5"].dropna() <= 0).all()

    def test_invalid_window_raises(self, sample_ohlcv_df):
        with pytest.raises(ValueError):
            create_rolling_window_mdd_feature(sample_ohlcv_df, window_size=1)

    def test_does_not_modify_original(self, sample_ohlcv_df):
        original = sample_ohlcv_df.copy()
        create_rolling_window_mdd_feature(sample_ohlcv_df)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)


class TestCreateRollingSharpeRatioFeature:
    def test_adds_sharpe_column(self, sample_ohlcv_df):
        result = create_rolling_sharpe_ratio_feature(sample_ohlcv_df, window_size=5)
        assert "sharpe_5" in result.columns

    def test_sharpe_values(self, sample_ohlcv_df):
        window = 5
        result = create_rolling_sharpe_ratio_feature(sample_ohlcv_df, window_size=window)
        for ticker in sample_ohlcv_df["Ticker"].unique():
            ticker_df = sample_ohlcv_df[sample_ohlcv_df["Ticker"] == ticker].sort_values("Date")
            prev = ticker_df["Close"].shift(1)
            daily_ret = (ticker_df["Close"] - prev) / prev
            mean_ret = daily_ret.rolling(window=window, min_periods=window).mean()
            std_ret = daily_ret.rolling(window=window, min_periods=window).std()
            expected = mean_ret / std_ret
            sharpe = result[result["Ticker"] == ticker].sort_values("Date")[f"sharpe_{window}"]
            valid = expected.dropna().index.intersection(sharpe.dropna().index)
            pd.testing.assert_series_equal(
                sharpe.loc[valid].reset_index(drop=True),
                expected.loc[valid].reset_index(drop=True),
                check_names=False,
                atol=1e-10,
            )

    def test_default_window(self, sample_ohlcv_df):
        result = create_rolling_sharpe_ratio_feature(sample_ohlcv_df)
        assert "sharpe_2" in result.columns

    def test_invalid_window_raises(self, sample_ohlcv_df):
        with pytest.raises(ValueError):
            create_rolling_sharpe_ratio_feature(sample_ohlcv_df, window_size=1)

    def test_does_not_modify_original(self, sample_ohlcv_df):
        original = sample_ohlcv_df.copy()
        create_rolling_sharpe_ratio_feature(sample_ohlcv_df)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)
