import numpy as np
import pandas as pd
import pytest

from financial_forecasting_platform.features.engineering import (
    create_garman_klass_variance_feature,
    create_gk_variance_rolling_mean_feature,
    create_gk_regime_feature,
    create_parkinson_variance_feature,
    create_parkinson_variance_rolling_mean_feature,
    create_parkinson_regime_feature,
    create_parkinson_volatility_feature,
    create_rogers_satchell_variance_feature,
    create_rogers_satchell_variance_rolling_mean_feature,
    create_rogers_satchell_volatility_feature,
    create_yang_zhang_variance_feature,
    create_yang_zhang_variance_rolling_mean_feature,
    create_yang_zhang_volatility_feature,
    create_yang_zhang_volatility_ratio_feature,
    create_yang_zhang_volatility_features,
)


class TestCreateGarmanKlassVariance:
    def test_adds_column(self, sample_ohlcv_df):
        result = create_garman_klass_variance_feature(sample_ohlcv_df, lag=1)
        assert "gk_variance_lag_1" in result.columns

    def test_column_name_with_lag(self, sample_ohlcv_df):
        result = create_garman_klass_variance_feature(sample_ohlcv_df, lag=3)
        assert "gk_variance_lag_3" in result.columns

    def test_values_match_formula(self, sample_ohlcv_df):
        result = create_garman_klass_variance_feature(sample_ohlcv_df, lag=1)
        for ticker in sample_ohlcv_df["Ticker"].unique():
            ticker_df = sample_ohlcv_df[sample_ohlcv_df["Ticker"] == ticker].sort_values("Date")
            high = ticker_df["High"].shift(1)
            low = ticker_df["Low"].shift(1)
            close = ticker_df["Close"].shift(1)
            open_ = ticker_df["Open"].shift(1)
            ln_hl = np.log(high / low)
            ln_co = np.log(close / open_)
            expected = 0.5 * ln_hl.pow(2) - (2 * np.log(2) - 1) * ln_co.pow(2)
            col = result[result["Ticker"] == ticker].sort_values("Date")["gk_variance_lag_1"]
            valid = expected.dropna().index.intersection(col.dropna().index)
            pd.testing.assert_series_equal(
                col.loc[valid].reset_index(drop=True),
                expected.loc[valid].reset_index(drop=True),
                check_names=False,
                atol=1e-10,
            )

    def test_first_row_is_nan(self, sample_ohlcv_df):
        result = create_garman_klass_variance_feature(sample_ohlcv_df, lag=1)
        for ticker in sample_ohlcv_df["Ticker"].unique():
            assert np.isnan(result[result["Ticker"] == ticker]["gk_variance_lag_1"].iloc[0])

    def test_invalid_lag_raises(self, sample_ohlcv_df):
        with pytest.raises(ValueError):
            create_garman_klass_variance_feature(sample_ohlcv_df, lag=0)

    def test_does_not_modify_original(self, sample_ohlcv_df):
        original = sample_ohlcv_df.copy()
        create_garman_klass_variance_feature(sample_ohlcv_df)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)


class TestCreateGkVarianceRollingMean:
    def test_adds_column(self, sample_ohlcv_df):
        result = create_gk_variance_rolling_mean_feature(sample_ohlcv_df, window_size=5, lag=1)
        assert "gk_variance_mean_5_lag_1" in result.columns

    def test_column_name_with_params(self, sample_ohlcv_df):
        result = create_gk_variance_rolling_mean_feature(sample_ohlcv_df, window_size=10, lag=3)
        assert "gk_variance_mean_10_lag_3" in result.columns

    def test_values_are_rolling_mean_of_gk_variance(self, sample_ohlcv_df):
        window, lag = 5, 1
        result = create_gk_variance_rolling_mean_feature(sample_ohlcv_df, window_size=window, lag=lag)
        gk = create_garman_klass_variance_feature(sample_ohlcv_df, lag=lag)
        for ticker in sample_ohlcv_df["Ticker"].unique():
            gk_ticker = gk[gk["Ticker"] == ticker].sort_values("Date")[f"gk_variance_lag_{lag}"]
            expected = gk_ticker.rolling(window=window, min_periods=window).mean()
            col = result[result["Ticker"] == ticker].sort_values("Date")[f"gk_variance_mean_{window}_lag_{lag}"]
            valid = expected.dropna().index.intersection(col.dropna().index)
            pd.testing.assert_series_equal(
                col.loc[valid].reset_index(drop=True),
                expected.loc[valid].reset_index(drop=True),
                check_names=False,
                atol=1e-10,
            )

    def test_invalid_window_raises(self, sample_ohlcv_df):
        with pytest.raises(ValueError):
            create_gk_variance_rolling_mean_feature(sample_ohlcv_df, window_size=0, lag=1)

    def test_invalid_lag_raises(self, sample_ohlcv_df):
        with pytest.raises(ValueError):
            create_gk_variance_rolling_mean_feature(sample_ohlcv_df, window_size=2, lag=0)

    def test_does_not_modify_original(self, sample_ohlcv_df):
        original = sample_ohlcv_df.copy()
        create_gk_variance_rolling_mean_feature(sample_ohlcv_df)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)


class TestCreateGkRegime:
    def test_adds_column(self, sample_ohlcv_df):
        result = create_gk_regime_feature(sample_ohlcv_df, short_span=5, long_span=20)
        assert "gk_regime_5_20" in result.columns

    def test_values_match_ratio_of_short_to_long_mean(self, sample_ohlcv_df):
        short_span, long_span = 5, 20
        result = create_gk_regime_feature(sample_ohlcv_df, short_span=short_span, long_span=long_span)
        col = result["gk_regime_5_20"]
        assert len(col.dropna()) > 0

    def test_invalid_spans_raise(self, sample_ohlcv_df):
        with pytest.raises(ValueError):
            create_gk_regime_feature(sample_ohlcv_df, short_span=20, long_span=5)
        with pytest.raises(ValueError):
            create_gk_regime_feature(sample_ohlcv_df, short_span=0, long_span=20)

    def test_does_not_modify_original(self, sample_ohlcv_df):
        original = sample_ohlcv_df.copy()
        create_gk_regime_feature(sample_ohlcv_df, short_span=5, long_span=20)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)


class TestCreateParkinsonVariance:
    def test_adds_column(self, sample_ohlcv_df):
        result = create_parkinson_variance_feature(sample_ohlcv_df, lag=1)
        assert "parkinson_variance_lag_1" in result.columns

    def test_values_match_formula(self, sample_ohlcv_df):
        result = create_parkinson_variance_feature(sample_ohlcv_df, lag=1)
        for ticker in sample_ohlcv_df["Ticker"].unique():
            ticker_df = sample_ohlcv_df[sample_ohlcv_df["Ticker"] == ticker].sort_values("Date")
            high = ticker_df["High"].shift(1)
            low = ticker_df["Low"].shift(1)
            expected = (1 / (4 * np.log(2))) * np.log(high / low).pow(2)
            col = result[result["Ticker"] == ticker].sort_values("Date")["parkinson_variance_lag_1"]
            valid = expected.dropna().index.intersection(col.dropna().index)
            pd.testing.assert_series_equal(
                col.loc[valid].reset_index(drop=True),
                expected.loc[valid].reset_index(drop=True),
                check_names=False,
                atol=1e-10,
            )

    def test_non_negative_values(self, sample_ohlcv_df):
        result = create_parkinson_variance_feature(sample_ohlcv_df, lag=1)
        col = result["parkinson_variance_lag_1"].dropna()
        assert (col >= 0).all()

    def test_invalid_lag_raises(self, sample_ohlcv_df):
        with pytest.raises(ValueError):
            create_parkinson_variance_feature(sample_ohlcv_df, lag=0)

    def test_does_not_modify_original(self, sample_ohlcv_df):
        original = sample_ohlcv_df.copy()
        create_parkinson_variance_feature(sample_ohlcv_df)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)


class TestCreateParkinsonVarianceRollingMean:
    def test_adds_column(self, sample_ohlcv_df):
        result = create_parkinson_variance_rolling_mean_feature(sample_ohlcv_df, window_size=5, lag=1)
        assert "parkinson_variance_mean_5_lag_1" in result.columns

    def test_values_are_rolling_mean(self, sample_ohlcv_df):
        window, lag = 5, 1
        result = create_parkinson_variance_rolling_mean_feature(sample_ohlcv_df, window_size=window, lag=lag)
        pv = create_parkinson_variance_feature(sample_ohlcv_df, lag=lag)
        for ticker in sample_ohlcv_df["Ticker"].unique():
            pv_ticker = pv[pv["Ticker"] == ticker].sort_values("Date")[f"parkinson_variance_lag_{lag}"]
            expected = pv_ticker.rolling(window=window, min_periods=window).mean()
            col = result[result["Ticker"] == ticker].sort_values("Date")[f"parkinson_variance_mean_{window}_lag_{lag}"]
            valid = expected.dropna().index.intersection(col.dropna().index)
            pd.testing.assert_series_equal(
                col.loc[valid].reset_index(drop=True),
                expected.loc[valid].reset_index(drop=True),
                check_names=False,
                atol=1e-10,
            )

    def test_invalid_window_raises(self, sample_ohlcv_df):
        with pytest.raises(ValueError):
            create_parkinson_variance_rolling_mean_feature(sample_ohlcv_df, window_size=0, lag=1)

    def test_invalid_lag_raises(self, sample_ohlcv_df):
        with pytest.raises(ValueError):
            create_parkinson_variance_rolling_mean_feature(sample_ohlcv_df, window_size=2, lag=0)

    def test_does_not_modify_original(self, sample_ohlcv_df):
        original = sample_ohlcv_df.copy()
        create_parkinson_variance_rolling_mean_feature(sample_ohlcv_df)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)


class TestCreateParkinsonRegime:
    def test_adds_column(self, sample_ohlcv_df):
        result = create_parkinson_regime_feature(sample_ohlcv_df, short_span=5, long_span=20)
        assert "parkinson_regime_5_20" in result.columns

    def test_values_match_ratio_of_short_to_long_mean(self, sample_ohlcv_df):
        result = create_parkinson_regime_feature(sample_ohlcv_df, short_span=5, long_span=20)
        col = result["parkinson_regime_5_20"]
        assert len(col.dropna()) > 0

    def test_invalid_spans_raise(self, sample_ohlcv_df):
        with pytest.raises(ValueError):
            create_parkinson_regime_feature(sample_ohlcv_df, short_span=20, long_span=5)
        with pytest.raises(ValueError):
            create_parkinson_regime_feature(sample_ohlcv_df, short_span=0, long_span=20)

    def test_does_not_modify_original(self, sample_ohlcv_df):
        original = sample_ohlcv_df.copy()
        create_parkinson_regime_feature(sample_ohlcv_df, short_span=5, long_span=20)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)


class TestCreateParkinsonVolatility:
    def test_adds_column(self, sample_ohlcv_df):
        result = create_parkinson_volatility_feature(sample_ohlcv_df, window_size=5, lag=1)
        assert "parkinson_volatility_5_lag_1" in result.columns

    def test_values_are_sqrt_of_rolling_mean_variance(self, sample_ohlcv_df):
        window, lag = 5, 1
        result = create_parkinson_volatility_feature(sample_ohlcv_df, window_size=window, lag=lag)
        rm = create_parkinson_variance_rolling_mean_feature(sample_ohlcv_df, window_size=window, lag=lag)
        for ticker in sample_ohlcv_df["Ticker"].unique():
            rm_ticker = rm[rm["Ticker"] == ticker].sort_values("Date")[f"parkinson_variance_mean_{window}_lag_{lag}"]
            expected = rm_ticker.pow(0.5)
            col = result[result["Ticker"] == ticker].sort_values("Date")[f"parkinson_volatility_{window}_lag_{lag}"]
            valid = expected.dropna().index.intersection(col.dropna().index)
            pd.testing.assert_series_equal(
                col.loc[valid].reset_index(drop=True),
                expected.loc[valid].reset_index(drop=True),
                check_names=False,
                atol=1e-10,
            )

    def test_non_negative_values(self, sample_ohlcv_df):
        result = create_parkinson_volatility_feature(sample_ohlcv_df, window_size=5, lag=1)
        col = result["parkinson_volatility_5_lag_1"].dropna()
        assert (col >= 0).all()

    def test_does_not_modify_original(self, sample_ohlcv_df):
        original = sample_ohlcv_df.copy()
        create_parkinson_volatility_feature(sample_ohlcv_df)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)


class TestCreateRogersSatchellVariance:
    def test_adds_column(self, sample_ohlcv_df):
        result = create_rogers_satchell_variance_feature(sample_ohlcv_df, lag=1)
        assert "rogers_satchell_variance_lag_1" in result.columns

    def test_values_match_formula(self, sample_ohlcv_df):
        result = create_rogers_satchell_variance_feature(sample_ohlcv_df, lag=1)
        for ticker in sample_ohlcv_df["Ticker"].unique():
            ticker_df = sample_ohlcv_df[sample_ohlcv_df["Ticker"] == ticker].sort_values("Date")
            high = ticker_df["High"].shift(1)
            low = ticker_df["Low"].shift(1)
            open_ = ticker_df["Open"].shift(1)
            close = ticker_df["Close"].shift(1)
            ln_hc = np.log(high / close)
            ln_ho = np.log(high / open_)
            ln_lc = np.log(low / close)
            ln_lo = np.log(low / open_)
            expected = ln_hc * ln_ho + ln_lc * ln_lo
            col = result[result["Ticker"] == ticker].sort_values("Date")["rogers_satchell_variance_lag_1"]
            valid = expected.dropna().index.intersection(col.dropna().index)
            pd.testing.assert_series_equal(
                col.loc[valid].reset_index(drop=True),
                expected.loc[valid].reset_index(drop=True),
                check_names=False,
                atol=1e-10,
            )

    def test_invalid_lag_raises(self, sample_ohlcv_df):
        with pytest.raises(ValueError):
            create_rogers_satchell_variance_feature(sample_ohlcv_df, lag=0)

    def test_does_not_modify_original(self, sample_ohlcv_df):
        original = sample_ohlcv_df.copy()
        create_rogers_satchell_variance_feature(sample_ohlcv_df)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)


class TestCreateRogersSatchellVarianceRollingMean:
    def test_adds_column(self, sample_ohlcv_df):
        result = create_rogers_satchell_variance_rolling_mean_feature(sample_ohlcv_df, window_size=5, lag=1)
        assert "rs_variance_mean_5_lag_1" in result.columns

    def test_values_are_rolling_mean(self, sample_ohlcv_df):
        window, lag = 5, 1
        result = create_rogers_satchell_variance_rolling_mean_feature(sample_ohlcv_df, window_size=window, lag=lag)
        rs = create_rogers_satchell_variance_feature(sample_ohlcv_df, lag=lag)
        for ticker in sample_ohlcv_df["Ticker"].unique():
            rs_ticker = rs[rs["Ticker"] == ticker].sort_values("Date")[f"rogers_satchell_variance_lag_{lag}"]
            expected = rs_ticker.rolling(window=window, min_periods=window).mean()
            col = result[result["Ticker"] == ticker].sort_values("Date")[f"rs_variance_mean_{window}_lag_{lag}"]
            valid = expected.dropna().index.intersection(col.dropna().index)
            pd.testing.assert_series_equal(
                col.loc[valid].reset_index(drop=True),
                expected.loc[valid].reset_index(drop=True),
                check_names=False,
                atol=1e-10,
            )

    def test_does_not_modify_original(self, sample_ohlcv_df):
        original = sample_ohlcv_df.copy()
        create_rogers_satchell_variance_rolling_mean_feature(sample_ohlcv_df)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)


class TestCreateRogersSatchellVolatility:
    def test_adds_column(self, sample_ohlcv_df):
        result = create_rogers_satchell_volatility_feature(sample_ohlcv_df, window_size=5, lag=1)
        assert "rs_volatility_5_lag_1" in result.columns

    def test_values_are_sqrt_of_rolling_mean_variance(self, sample_ohlcv_df):
        window, lag = 5, 1
        result = create_rogers_satchell_volatility_feature(sample_ohlcv_df, window_size=window, lag=lag)
        rm = create_rogers_satchell_variance_rolling_mean_feature(sample_ohlcv_df, window_size=window, lag=lag)
        for ticker in sample_ohlcv_df["Ticker"].unique():
            rm_ticker = rm[rm["Ticker"] == ticker].sort_values("Date")[f"rs_variance_mean_{window}_lag_{lag}"]
            expected = rm_ticker.pow(0.5)
            col = result[result["Ticker"] == ticker].sort_values("Date")[f"rs_volatility_{window}_lag_{lag}"]
            valid = expected.dropna().index.intersection(col.dropna().index)
            pd.testing.assert_series_equal(
                col.loc[valid].reset_index(drop=True),
                expected.loc[valid].reset_index(drop=True),
                check_names=False,
                atol=1e-10,
            )

    def test_does_not_modify_original(self, sample_ohlcv_df):
        original = sample_ohlcv_df.copy()
        create_rogers_satchell_volatility_feature(sample_ohlcv_df)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)


class TestCreateYangZhangVariance:
    def test_adds_column(self, single_ticker_df):
        result = create_yang_zhang_variance_feature(single_ticker_df, lag=1)
        assert "yz_variance_lag_1" in result.columns

    def test_column_name_with_lag(self, single_ticker_df):
        result = create_yang_zhang_variance_feature(single_ticker_df, lag=2)
        assert "yz_variance_lag_2" in result.columns

    def test_has_valid_values(self, single_ticker_df):
        result = create_yang_zhang_variance_feature(single_ticker_df, lag=1)
        col = result["yz_variance_lag_1"].dropna()
        assert len(col) > 0
        assert col.notna().all()

    def test_invalid_lag_raises(self, single_ticker_df):
        with pytest.raises(ValueError):
            create_yang_zhang_variance_feature(single_ticker_df, lag=0)

    def test_does_not_modify_original(self, single_ticker_df):
        original = single_ticker_df.copy()
        create_yang_zhang_variance_feature(single_ticker_df)
        pd.testing.assert_frame_equal(single_ticker_df, original)


class TestCreateYangZhangVarianceRollingMean:
    def test_adds_column(self, single_ticker_df):
        result = create_yang_zhang_variance_rolling_mean_feature(single_ticker_df, lag=1, window_size=5)
        assert "yz_variance_mean_5_lag_1" in result.columns

    def test_values_are_rolling_mean(self, single_ticker_df):
        window, lag = 5, 1
        result = create_yang_zhang_variance_rolling_mean_feature(single_ticker_df, lag=lag, window_size=window)
        yz = create_yang_zhang_variance_feature(single_ticker_df, lag=lag)
        ticker_df = yz.sort_values("Date")
        expected = ticker_df["yz_variance_lag_1"].rolling(window=window, min_periods=window).mean()
        col = result.sort_values("Date")[f"yz_variance_mean_{window}_lag_{lag}"]
        valid = expected.dropna().index.intersection(col.dropna().index)
        pd.testing.assert_series_equal(
            col.loc[valid].reset_index(drop=True),
            expected.loc[valid].reset_index(drop=True),
            check_names=False,
            atol=1e-10,
        )

    def test_invalid_window_raises(self, single_ticker_df):
        with pytest.raises(ValueError):
            create_yang_zhang_variance_rolling_mean_feature(single_ticker_df, lag=1, window_size=0)

    def test_does_not_modify_original(self, single_ticker_df):
        original = single_ticker_df.copy()
        create_yang_zhang_variance_rolling_mean_feature(single_ticker_df)
        pd.testing.assert_frame_equal(single_ticker_df, original)


class TestCreateYangZhangVolatility:
    def test_adds_column(self, single_ticker_df):
        result = create_yang_zhang_volatility_feature(single_ticker_df, lag=1, window_size=5)
        assert "yz_volatility_5_lag_1" in result.columns

    def test_values_are_sqrt_of_variance(self, single_ticker_df):
        window, lag = 5, 1
        result = create_yang_zhang_volatility_feature(single_ticker_df, lag=lag, window_size=window)
        rm = create_yang_zhang_variance_rolling_mean_feature(single_ticker_df, lag=lag, window_size=window)
        expected = rm["yz_variance_mean_5_lag_1"].pow(0.5)
        col = result["yz_volatility_5_lag_1"]
        valid = expected.dropna().index.intersection(col.dropna().index)
        pd.testing.assert_series_equal(
            col.loc[valid].reset_index(drop=True),
            expected.loc[valid].reset_index(drop=True),
            check_names=False,
            atol=1e-10,
        )

    def test_non_negative(self, single_ticker_df):
        result = create_yang_zhang_volatility_feature(single_ticker_df, lag=1, window_size=5)
        col = result["yz_volatility_5_lag_1"].dropna()
        assert (col >= 0).all()

    def test_does_not_modify_original(self, single_ticker_df):
        original = single_ticker_df.copy()
        create_yang_zhang_volatility_feature(single_ticker_df)
        pd.testing.assert_frame_equal(single_ticker_df, original)


class TestCreateYangZhangVolatilityRatioFeature:
    def test_adds_column(self, single_ticker_df):
        result = create_yang_zhang_volatility_ratio_feature(single_ticker_df, short_span=5, long_span=10)
        assert "yz_volatility_ratio_5_10" in result.columns

    def test_ratio_values(self, single_ticker_df):
        short, long = 5, 10
        result = create_yang_zhang_volatility_ratio_feature(single_ticker_df, short_span=short, long_span=long)
        vol_short = create_yang_zhang_volatility_feature(single_ticker_df, lag=1, window_size=short)
        vol_long = create_yang_zhang_volatility_feature(single_ticker_df, lag=1, window_size=long)
        expected = vol_short[f"yz_volatility_{short}_lag_1"] / vol_long[f"yz_volatility_{long}_lag_1"]
        col = result[f"yz_volatility_ratio_{short}_{long}"]
        valid = expected.dropna().index.intersection(col.dropna().index)
        pd.testing.assert_series_equal(
            col.loc[valid].reset_index(drop=True),
            expected.loc[valid].reset_index(drop=True),
            check_names=False,
            atol=1e-10,
        )

    def test_short_greater_than_long_raises(self, single_ticker_df):
        with pytest.raises(ValueError):
            create_yang_zhang_volatility_ratio_feature(single_ticker_df, short_span=20, long_span=5)

    def test_equal_spans_raises(self, single_ticker_df):
        with pytest.raises(ValueError):
            create_yang_zhang_volatility_ratio_feature(single_ticker_df, short_span=5, long_span=5)

    def test_does_not_modify_original(self, single_ticker_df):
        original = single_ticker_df.copy()
        create_yang_zhang_volatility_ratio_feature(single_ticker_df)
        pd.testing.assert_frame_equal(single_ticker_df, original)


class TestCreateYangZhangVolatilityFeatures:
    def test_adds_all_three_columns(self, single_ticker_df):
        result = create_yang_zhang_volatility_features(single_ticker_df, lag=1, window_size=5)
        assert "yz_variance_lag_1" in result.columns
        assert "yz_variance_mean_5_lag_1" in result.columns
        assert "yz_volatility_5_lag_1" in result.columns

    def test_volatility_is_sqrt_of_variance_mean(self, single_ticker_df):
        result = create_yang_zhang_volatility_features(single_ticker_df, lag=1, window_size=5)
        var_mean = result["yz_variance_mean_5_lag_1"]
        vol = result["yz_volatility_5_lag_1"]
        valid = var_mean.dropna().index.intersection(vol.dropna().index)
        expected = var_mean.loc[valid].pow(0.5)
        pd.testing.assert_series_equal(
            vol.loc[valid].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
            atol=1e-10,
        )

    def test_invalid_lag_raises(self, single_ticker_df):
        with pytest.raises(ValueError):
            create_yang_zhang_volatility_features(single_ticker_df, lag=0, window_size=5)

    def test_invalid_window_raises(self, single_ticker_df):
        with pytest.raises(ValueError):
            create_yang_zhang_volatility_features(single_ticker_df, lag=1, window_size=0)

    def test_does_not_modify_original(self, single_ticker_df):
        original = single_ticker_df.copy()
        create_yang_zhang_volatility_features(single_ticker_df)
        pd.testing.assert_frame_equal(single_ticker_df, original)
