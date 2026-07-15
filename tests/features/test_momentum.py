import numpy as np
import pandas as pd
import pytest

from financial_forecasting_platform.features.engineering import create_rsi_feature


class TestCreateRsiFeature:
    def test_adds_rsi_column(self, sample_ohlcv_df):
        result = create_rsi_feature(sample_ohlcv_df, window_size=14)
        assert "rsi_14" in result.columns

    def test_rsi_bounded_between_0_and_100(self, sample_ohlcv_df):
        result = create_rsi_feature(sample_ohlcv_df, window_size=14)
        rsi = result["rsi_14"].dropna()
        assert (rsi >= 0).all()
        assert (rsi <= 100).all()

    def test_rsi_custom_window(self, sample_ohlcv_df):
        result = create_rsi_feature(sample_ohlcv_df, window_size=5)
        assert "rsi_5" in result.columns
        rsi = result["rsi_5"].dropna()
        assert (rsi >= 0).all()
        assert (rsi <= 100).all()

    def test_rsi_default_window(self, sample_ohlcv_df):
        result = create_rsi_feature(sample_ohlcv_df)
        assert "rsi_14" in result.columns

    def test_rsi_per_ticker(self, sample_ohlcv_df):
        result = create_rsi_feature(sample_ohlcv_df, window_size=14)
        for ticker in sample_ohlcv_df["Ticker"].unique():
            ticker_result = result[result["Ticker"] == ticker]
            rsi = ticker_result["rsi_14"].dropna()
            assert len(rsi) > 0
            assert (rsi >= 0).all()
            assert (rsi <= 100).all()

    def test_monotonically_increasing_price_gives_high_rsi(self):
        dates = pd.date_range("2024-01-01", periods=30, freq="B")
        close = np.arange(100, 130, dtype=float)
        df = pd.DataFrame(
            {
                "Date": dates,
                "Ticker": "TEST",
                "Open": close - 1,
                "High": close + 1,
                "Low": close - 2,
                "Close": close,
                "Volume": [1_000_000] * len(close),
            },
        )
        result = create_rsi_feature(df, window_size=14)
        late_rsi = result.loc[result["Ticker"] == "TEST", "rsi_14"].dropna()
        assert late_rsi.iloc[-1] > 70

    def test_invalid_window_raises(self, sample_ohlcv_df):
        with pytest.raises(ValueError):
            create_rsi_feature(sample_ohlcv_df, window_size=1)

    def test_does_not_modify_original(self, sample_ohlcv_df):
        original = sample_ohlcv_df.copy()
        create_rsi_feature(sample_ohlcv_df)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)
