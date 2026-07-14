import numpy as np
import pandas as pd
import pytest

from financial_forecasting_platform.features.engineering import (
    create_body_percentage_feature,
    create_candle_body_feature,
    create_daily_range_feature,
    create_lower_shadow_feature,
    create_lower_shadow_percentage_feature,
    create_range_percentage_feature,
    create_upper_shadow_feature,
    create_upper_shadow_percentage_feature,
)


class TestCreateDailyRangeFeature:
    def test_adds_daily_range(self, sample_ohlcv_df):
        result = create_daily_range_feature(sample_ohlcv_df)
        assert "daily_range" in result.columns

    def test_daily_range_values(self, sample_ohlcv_df):
        result = create_daily_range_feature(sample_ohlcv_df)
        expected = sample_ohlcv_df["High"] - sample_ohlcv_df["Low"]
        pd.testing.assert_series_equal(
            result["daily_range"].sort_index(),
            expected.sort_index(),
            check_names=False,
        )

    def test_daily_range_non_negative(self, sample_ohlcv_df):
        result = create_daily_range_feature(sample_ohlcv_df)
        assert (result["daily_range"] >= 0).all()

    def test_does_not_modify_original(self, sample_ohlcv_df):
        original = sample_ohlcv_df.copy()
        create_daily_range_feature(sample_ohlcv_df)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)


class TestCreateRangePercentageFeature:
    def test_adds_range_percentage(self, sample_ohlcv_df):
        result = create_range_percentage_feature(sample_ohlcv_df)
        assert "range_percentage" in result.columns

    def test_range_percentage_values(self, sample_ohlcv_df):
        result = create_range_percentage_feature(sample_ohlcv_df)
        expected = (sample_ohlcv_df["High"] - sample_ohlcv_df["Low"]) / sample_ohlcv_df[
            "Close"
        ]
        pd.testing.assert_series_equal(
            result["range_percentage"].sort_index(),
            expected.sort_index(),
            check_names=False,
            atol=1e-10,
        )

    def test_does_not_modify_original(self, sample_ohlcv_df):
        original = sample_ohlcv_df.copy()
        create_range_percentage_feature(sample_ohlcv_df)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)


class TestCreateCandleBodyFeature:
    def test_adds_candle_body(self, sample_ohlcv_df):
        result = create_candle_body_feature(sample_ohlcv_df)
        assert "candle_body" in result.columns

    def test_candle_body_values(self, sample_ohlcv_df):
        result = create_candle_body_feature(sample_ohlcv_df)
        expected = sample_ohlcv_df["Close"] - sample_ohlcv_df["Open"]
        pd.testing.assert_series_equal(
            result["candle_body"].sort_index(),
            expected.sort_index(),
            check_names=False,
        )

    def test_does_not_modify_original(self, sample_ohlcv_df):
        original = sample_ohlcv_df.copy()
        create_candle_body_feature(sample_ohlcv_df)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)


class TestCreateBodyPercentageFeature:
    def test_adds_body_percentage(self, sample_ohlcv_df):
        result = create_body_percentage_feature(sample_ohlcv_df)
        assert "body_percentage" in result.columns

    def test_body_percentage_values(self, sample_ohlcv_df):
        result = create_body_percentage_feature(sample_ohlcv_df)
        expected = (sample_ohlcv_df["Close"] - sample_ohlcv_df["Open"]) / sample_ohlcv_df[
            "Open"
        ]
        pd.testing.assert_series_equal(
            result["body_percentage"].sort_index(),
            expected.sort_index(),
            check_names=False,
            atol=1e-10,
        )

    def test_does_not_modify_original(self, sample_ohlcv_df):
        original = sample_ohlcv_df.copy()
        create_body_percentage_feature(sample_ohlcv_df)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)


class TestCreateUpperShadowFeature:
    def test_adds_upper_shadow(self, sample_ohlcv_df):
        result = create_upper_shadow_feature(sample_ohlcv_df)
        assert "upper_shadow" in result.columns

    def test_upper_shadow_values(self, sample_ohlcv_df):
        result = create_upper_shadow_feature(sample_ohlcv_df)
        expected = sample_ohlcv_df["High"] - sample_ohlcv_df[["Open", "Close"]].max(axis=1)
        pd.testing.assert_series_equal(
            result["upper_shadow"].sort_index(),
            expected.sort_index(),
            check_names=False,
        )

    def test_upper_shadow_non_negative(self, sample_ohlcv_df):
        result = create_upper_shadow_feature(sample_ohlcv_df)
        assert (result["upper_shadow"] >= 0).all()

    def test_does_not_modify_original(self, sample_ohlcv_df):
        original = sample_ohlcv_df.copy()
        create_upper_shadow_feature(sample_ohlcv_df)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)


class TestCreateLowerShadowFeature:
    def test_adds_lower_shadow(self, sample_ohlcv_df):
        result = create_lower_shadow_feature(sample_ohlcv_df)
        assert "lower_shadow" in result.columns

    def test_lower_shadow_values(self, sample_ohlcv_df):
        result = create_lower_shadow_feature(sample_ohlcv_df)
        expected = sample_ohlcv_df[["Open", "Close"]].min(axis=1) - sample_ohlcv_df["Low"]
        pd.testing.assert_series_equal(
            result["lower_shadow"].sort_index(),
            expected.sort_index(),
            check_names=False,
        )

    def test_lower_shadow_non_negative(self, sample_ohlcv_df):
        result = create_lower_shadow_feature(sample_ohlcv_df)
        assert (result["lower_shadow"] >= 0).all()

    def test_does_not_modify_original(self, sample_ohlcv_df):
        original = sample_ohlcv_df.copy()
        create_lower_shadow_feature(sample_ohlcv_df)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)


class TestCreateUpperShadowPercentageFeature:
    def test_adds_upper_shadow_pct(self, sample_ohlcv_df):
        result = create_upper_shadow_percentage_feature(sample_ohlcv_df)
        assert "upper_shadow_pct" in result.columns

    def test_upper_shadow_pct_values(self, sample_ohlcv_df):
        result = create_upper_shadow_percentage_feature(sample_ohlcv_df)
        upper_shadow = sample_ohlcv_df["High"] - sample_ohlcv_df[["Open", "Close"]].max(
            axis=1
        )
        expected = upper_shadow / sample_ohlcv_df["Close"]
        pd.testing.assert_series_equal(
            result["upper_shadow_pct"].sort_index(),
            expected.sort_index(),
            check_names=False,
            atol=1e-10,
        )

    def test_does_not_modify_original(self, sample_ohlcv_df):
        original = sample_ohlcv_df.copy()
        create_upper_shadow_percentage_feature(sample_ohlcv_df)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)


class TestCreateLowerShadowPercentageFeature:
    def test_adds_lower_shadow_pct(self, sample_ohlcv_df):
        result = create_lower_shadow_percentage_feature(sample_ohlcv_df)
        assert "lower_shadow_pct" in result.columns

    def test_lower_shadow_pct_values(self, sample_ohlcv_df):
        result = create_lower_shadow_percentage_feature(sample_ohlcv_df)
        lower_shadow = sample_ohlcv_df[["Open", "Close"]].min(axis=1) - sample_ohlcv_df[
            "Low"
        ]
        expected = lower_shadow / sample_ohlcv_df["Close"]
        pd.testing.assert_series_equal(
            result["lower_shadow_pct"].sort_index(),
            expected.sort_index(),
            check_names=False,
            atol=1e-10,
        )

    def test_does_not_modify_original(self, sample_ohlcv_df):
        original = sample_ohlcv_df.copy()
        create_lower_shadow_percentage_feature(sample_ohlcv_df)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)
