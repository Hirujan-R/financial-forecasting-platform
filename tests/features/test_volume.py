import numpy as np
import pandas as pd
import pytest

from financial_forecasting_platform.features.engineering import (
    create_relative_volume_feature,
    create_return_x_volume_feature,
    create_volume_pct_change_feature,
    create_volume_sma_feature,
)


class TestCreateVolumePctChangeFeature:
    def test_adds_volume_pct_change(self, sample_ohlcv_df):
        result = create_volume_pct_change_feature(sample_ohlcv_df, lag=1)
        assert "volume_pct_change" in result.columns

    def test_volume_pct_change_values(self, sample_ohlcv_df):
        result = create_volume_pct_change_feature(sample_ohlcv_df, lag=1)
        for ticker in sample_ohlcv_df["Ticker"].unique():
            ticker_df = sample_ohlcv_df[sample_ohlcv_df["Ticker"] == ticker].sort_values("Date")
            vol = result[result["Ticker"] == ticker].sort_values("Date")["volume_pct_change"]
            prev_vol = ticker_df["Volume"].shift(1)
            expected = (ticker_df["Volume"] - prev_vol) / prev_vol
            valid = expected.dropna().index.intersection(vol.dropna().index)
            pd.testing.assert_series_equal(
                vol.loc[valid].reset_index(drop=True),
                expected.loc[valid].reset_index(drop=True),
                check_names=False,
                atol=1e-10,
            )

    def test_first_row_per_ticker_is_nan(self, sample_ohlcv_df):
        result = create_volume_pct_change_feature(sample_ohlcv_df, lag=1)
        for ticker in sample_ohlcv_df["Ticker"].unique():
            ticker_result = result[result["Ticker"] == ticker]
            assert np.isnan(ticker_result["volume_pct_change"].iloc[0])

    def test_invalid_lag_raises(self, sample_ohlcv_df):
        with pytest.raises(ValueError):
            create_volume_pct_change_feature(sample_ohlcv_df, lag=0)

    def test_does_not_modify_original(self, sample_ohlcv_df):
        original = sample_ohlcv_df.copy()
        create_volume_pct_change_feature(sample_ohlcv_df)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)


class TestCreateVolumeSmaFeature:
    def test_adds_volume_sma_column(self, sample_ohlcv_df):
        result = create_volume_sma_feature(sample_ohlcv_df, window_size=5)
        assert "volume_sma_5" in result.columns

    def test_volume_sma_values(self, sample_ohlcv_df):
        window = 5
        result = create_volume_sma_feature(sample_ohlcv_df, window_size=window)
        for ticker in sample_ohlcv_df["Ticker"].unique():
            ticker_df = sample_ohlcv_df[sample_ohlcv_df["Ticker"] == ticker].sort_values("Date")
            expected = ticker_df["Volume"].rolling(
                window=window, min_periods=window
            ).mean()
            sma = result[result["Ticker"] == ticker].sort_values("Date")[f"volume_sma_{window}"]
            valid = expected.dropna().index.intersection(sma.dropna().index)
            pd.testing.assert_series_equal(
                sma.loc[valid].reset_index(drop=True),
                expected.loc[valid].reset_index(drop=True),
                check_names=False,
                atol=1e-10,
            )

    def test_invalid_window_raises(self, sample_ohlcv_df):
        with pytest.raises(ValueError):
            create_volume_sma_feature(sample_ohlcv_df, window_size=0)

    def test_does_not_modify_original(self, sample_ohlcv_df):
        original = sample_ohlcv_df.copy()
        create_volume_sma_feature(sample_ohlcv_df)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)


class TestCreateRelativeVolumeFeature:
    def test_adds_relative_volume(self, sample_ohlcv_df):
        result = create_relative_volume_feature(sample_ohlcv_df, window_size=5)
        assert "relative_volume_5" in result.columns

    def test_relative_volume_values(self, sample_ohlcv_df):
        window = 5
        result = create_relative_volume_feature(sample_ohlcv_df, window_size=window)
        for ticker in sample_ohlcv_df["Ticker"].unique():
            ticker_df = sample_ohlcv_df[sample_ohlcv_df["Ticker"] == ticker].sort_values("Date")
            avg_vol = ticker_df["Volume"].rolling(
                window=window, min_periods=window
            ).mean()
            expected = ticker_df["Volume"] / avg_vol
            rel = result[result["Ticker"] == ticker].sort_values("Date")[f"relative_volume_{window}"]
            valid = expected.dropna().index.intersection(rel.dropna().index)
            pd.testing.assert_series_equal(
                rel.loc[valid].reset_index(drop=True),
                expected.loc[valid].reset_index(drop=True),
                check_names=False,
                atol=1e-10,
            )

    def test_relative_volume_positive(self, sample_ohlcv_df):
        result = create_relative_volume_feature(sample_ohlcv_df, window_size=5)
        assert (result["relative_volume_5"].dropna() > 0).all()

    def test_does_not_modify_original(self, sample_ohlcv_df):
        original = sample_ohlcv_df.copy()
        create_relative_volume_feature(sample_ohlcv_df)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)


class TestCreateReturnXVolumeFeature:
    def test_adds_return_x_volume(self, sample_ohlcv_df):
        result = create_return_x_volume_feature(sample_ohlcv_df)
        assert "return_x_volume" in result.columns

    def test_return_x_volume_values(self, sample_ohlcv_df):
        result = create_return_x_volume_feature(sample_ohlcv_df)
        for ticker in sample_ohlcv_df["Ticker"].unique():
            ticker_df = sample_ohlcv_df[sample_ohlcv_df["Ticker"] == ticker].sort_values("Date")
            price_return = ticker_df["Close"].diff()
            expected = ticker_df["Volume"] * price_return
            col = result[result["Ticker"] == ticker].sort_values("Date")["return_x_volume"]
            valid = expected.dropna().index.intersection(col.dropna().index)
            pd.testing.assert_series_equal(
                col.loc[valid].reset_index(drop=True),
                expected.loc[valid].reset_index(drop=True),
                check_names=False,
                atol=1e-10,
            )

    def test_does_not_modify_original(self, sample_ohlcv_df):
        original = sample_ohlcv_df.copy()
        create_return_x_volume_feature(sample_ohlcv_df)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)
