import numpy as np
import pandas as pd
import pytest
from kedro.pipeline import Pipeline

from financial_forecasting_platform.pipelines.feature_engineering.nodes import (
    feature_engineering,
    target_variable_engineering,
)
from financial_forecasting_platform.pipelines.feature_engineering.pipeline import (
    create_pipeline,
)


@pytest.fixture
def ohlcv_df():
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=60, freq="B")
    rows = []
    for ticker in ["AAPL", "MSFT"]:
        close = 150 + np.cumsum(np.random.randn(60) * 2)
        rows.append(
            pd.DataFrame(
                {
                    "Date": dates,
                    "Ticker": ticker,
                    "Open": close + np.random.randn(60) * 0.5,
                    "High": close + abs(np.random.randn(60)),
                    "Low": close - abs(np.random.randn(60)),
                    "Close": close,
                    "Volume": np.random.randint(1_000_000, 10_000_000, size=60),
                }
            )
        )
    df = pd.concat(rows, ignore_index=True)
    df["High"] = df[["Open", "Close", "High"]].max(axis=1)
    df["Low"] = df[["Open", "Close", "Low"]].min(axis=1)
    return df


class TestFeatureEngineering:
    def test_returns_dataframe(self, ohlcv_df):
        result = feature_engineering(ohlcv_df)
        assert isinstance(result, pd.DataFrame)

    def test_preserves_row_count(self, ohlcv_df):
        result = feature_engineering(ohlcv_df)
        assert len(result) == len(ohlcv_df)

    def test_preserves_tickers(self, ohlcv_df):
        result = feature_engineering(ohlcv_df)
        assert set(result["Ticker"].unique()) == {"AAPL", "MSFT"}

    def test_does_not_modify_original(self, ohlcv_df):
        original = ohlcv_df.copy()
        feature_engineering(ohlcv_df)
        pd.testing.assert_frame_equal(ohlcv_df, original)

    def test_has_return_features(self, ohlcv_df):
        result = feature_engineering(ohlcv_df)
        for lag in [1, 5, 10, 20]:
            assert f"return_lag_{lag}" in result.columns
        assert "log_return_lag_1" in result.columns

    def test_has_trend_features(self, ohlcv_df):
        result = feature_engineering(ohlcv_df)
        assert "SMA_5" in result.columns
        assert "SMA_20" in result.columns
        assert "SMA_50" in result.columns
        assert "ema_12" in result.columns
        assert "ema_26" in result.columns
        assert "close_vs_SMA_20" in result.columns

    def test_has_momentum_features(self, ohlcv_df):
        result = feature_engineering(ohlcv_df)
        assert "rsi_14" in result.columns
        assert "momentum_5" in result.columns
        assert "momentum_20" in result.columns
        assert "momentum_60" in result.columns

    def test_has_volatility_features(self, ohlcv_df):
        result = feature_engineering(ohlcv_df)
        for w in [5, 20, 60]:
            assert f"simple_return_volatility_{w}" in result.columns
            assert f"log_return_volatility_{w}" in result.columns
        assert "bollinger_upper_distance_30" in result.columns
        assert "bollinger_lower_distance_30" in result.columns
        assert "bollinger_bandwidth_30" in result.columns

    def test_has_price_action_features(self, ohlcv_df):
        result = feature_engineering(ohlcv_df)
        assert "daily_range" in result.columns
        assert "range_percentage" in result.columns
        assert "candle_body" in result.columns
        assert "body_percentage" in result.columns
        assert "upper_shadow" in result.columns
        assert "lower_shadow" in result.columns
        assert "upper_shadow_pct" in result.columns
        assert "lower_shadow_pct" in result.columns

    def test_has_volume_features(self, ohlcv_df):
        result = feature_engineering(ohlcv_df)
        assert "volume_pct_change" in result.columns
        assert "relative_volume_30" in result.columns
        assert "volume_sma_20" in result.columns
        assert "return_x_volume" in result.columns

    def test_has_risk_features(self, ohlcv_df):
        result = feature_engineering(ohlcv_df)
        assert "drawdown_30" in result.columns
        assert "rolling_window_mdd_30" in result.columns
        assert "sharpe_30" in result.columns

    def test_has_date_features(self, ohlcv_df):
        result = feature_engineering(ohlcv_df)
        assert "day_of_week" in result.columns
        assert "month" in result.columns

    def test_has_original_ohlcv_columns(self, ohlcv_df):
        result = feature_engineering(ohlcv_df)
        for col in ["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]:
            assert col in result.columns

    def test_total_expected_column_count(self, ohlcv_df):
        result = feature_engineering(ohlcv_df)
        assert result.shape[1] >= 45


class TestTargetVariableEngineering:
    def test_returns_series(self, ohlcv_df):
        result = target_variable_engineering(ohlcv_df)
        assert isinstance(result, pd.Series)

    def test_length_matches_input(self, ohlcv_df):
        result = target_variable_engineering(ohlcv_df)
        assert len(result) == len(ohlcv_df)

    def test_values_are_binary_or_nan(self, ohlcv_df):
        result = target_variable_engineering(ohlcv_df)
        non_nan = result.dropna()
        assert set(non_nan.unique()).issubset({0.0, 1.0})

    def test_series_name(self, ohlcv_df):
        result = target_variable_engineering(ohlcv_df)
        assert result.name == "market_movement"


class TestFeatureEngineeringPipeline:
    def test_returns_pipeline(self):
        result = create_pipeline()
        assert isinstance(result, Pipeline)

    def test_pipeline_node_name(self):
        pipeline = create_pipeline()
        node_names = [n.name for n in pipeline.nodes]
        assert "feature_engineering_node" in node_names

    def test_pipeline_inputs_cleaned_data(self):
        pipeline = create_pipeline()
        input_datasets = set()
        for n in pipeline.nodes:
            input_datasets.update(n.inputs)
        assert "validated_raw_data" in input_datasets

    def test_pipeline_outputs_engineered_data(self):
        pipeline = create_pipeline()
        output_datasets = set()
        for n in pipeline.nodes:
            output_datasets.update(n.outputs)
        assert "engineered_data" in output_datasets

    def test_pipeline_has_one_node(self):
        pipeline = create_pipeline()
        assert len(pipeline.nodes) == 1
