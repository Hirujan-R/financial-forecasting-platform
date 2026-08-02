import numpy as np
import pandas as pd
import pytest
from kedro.pipeline import Pipeline

from financial_forecasting_platform.pipelines.feature_engineering.nodes import (
    create_all_features,
    lr_feature_engineering,
    xgboost_feature_engineering,
    mlp_feature_engineering,
    FEATURE_FUNCTION_REGISTRY,
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


@pytest.fixture
def feature_config():
    return [
        {"function": "create_lag_return_feature", "kwargs": {"lag": 1}},
        {"function": "create_simple_moving_average_feature", "kwargs": {"window_size": 20}},
        {"function": "create_date_features", "kwargs": {}},
        {"function": "create_market_movement_target", "kwargs": {"forward_window": 5, "lookback_window": 20}},
    ]


class TestCreateAllFeatures:
    def test_returns_dataframe(self, ohlcv_df, feature_config):
        result = create_all_features(ohlcv_df, feature_config)
        assert isinstance(result, pd.DataFrame)

    def test_preserves_row_count(self, ohlcv_df, feature_config):
        result = create_all_features(ohlcv_df, feature_config)
        assert len(result) == len(ohlcv_df)

    def test_does_not_modify_original(self, ohlcv_df, feature_config):
        original = ohlcv_df.copy()
        create_all_features(ohlcv_df, feature_config)
        pd.testing.assert_frame_equal(ohlcv_df, original)

    def test_drops_ohlcv_columns_by_default(self, ohlcv_df, feature_config):
        result = create_all_features(ohlcv_df, feature_config)
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            assert col not in result.columns

    def test_sets_date_as_index(self, ohlcv_df, feature_config):
        result = create_all_features(ohlcv_df, feature_config)
        assert result.index.name == "Date"

    def test_invalid_function_name_raises(self, ohlcv_df):
        bad_config = [{"function": "nonexistent_function", "kwargs": {}}]
        with pytest.raises(ValueError, match="not registered"):
            create_all_features(ohlcv_df, bad_config)

    def test_creates_configured_features(self, ohlcv_df, feature_config):
        result = create_all_features(ohlcv_df, feature_config)
        assert "return_lag_1" in result.columns
        assert "SMA_20" in result.columns
        assert "day_of_week" in result.columns
        assert "market_movement" in result.columns

    def test_custom_columns_to_drop(self, ohlcv_df, feature_config):
        result = create_all_features(ohlcv_df, feature_config, columns_to_drop=["Open", "Close"])
        assert "Open" not in result.columns
        assert "Close" not in result.columns
        assert "High" in result.columns


class TestFeatureFunctionRegistry:
    def test_registry_is_not_empty(self):
        assert len(FEATURE_FUNCTION_REGISTRY) > 0

    def test_registry_contains_expected_functions(self):
        assert "create_lag_return_feature" in FEATURE_FUNCTION_REGISTRY
        assert "create_simple_moving_average_feature" in FEATURE_FUNCTION_REGISTRY
        assert "create_market_movement_target" in FEATURE_FUNCTION_REGISTRY
        assert "create_rsi_feature" in FEATURE_FUNCTION_REGISTRY

    def test_registry_excludes_create_all_features(self):
        assert "create_all_features" not in FEATURE_FUNCTION_REGISTRY


class TestLrFeatureEngineering:
    def test_returns_dataframe(self, ohlcv_df):
        custom = ["Ticker", "Close", "Volume"]
        result = lr_feature_engineering(ohlcv_df, features=custom)
        assert isinstance(result, pd.DataFrame)

    def test_selects_specified_columns(self, ohlcv_df):
        custom = ["Ticker", "Close", "Volume", "Open"]
        result = lr_feature_engineering(ohlcv_df, features=custom)
        assert list(result.columns) == custom

    def test_preserves_row_count(self, ohlcv_df):
        custom = ["Ticker", "Close"]
        result = lr_feature_engineering(ohlcv_df, features=custom)
        assert len(result) == len(ohlcv_df)

    def test_does_not_modify_original(self, ohlcv_df):
        original = ohlcv_df.copy()
        lr_feature_engineering(ohlcv_df, features=["Ticker", "Close"])
        pd.testing.assert_frame_equal(ohlcv_df, original)

    def test_missing_column_raises(self, ohlcv_df):
        with pytest.raises(KeyError):
            lr_feature_engineering(ohlcv_df, features=["Ticker", "nonexistent_col"])


class TestXgboostFeatureEngineering:
    def test_returns_dataframe(self, ohlcv_df):
        custom = ["Ticker", "Close", "Volume"]
        result = xgboost_feature_engineering(ohlcv_df, features=custom)
        assert isinstance(result, pd.DataFrame)

    def test_selects_specified_columns(self, ohlcv_df):
        custom = ["Ticker", "Close", "Volume", "Open"]
        result = xgboost_feature_engineering(ohlcv_df, features=custom)
        assert list(result.columns) == custom

    def test_missing_column_raises(self, ohlcv_df):
        with pytest.raises(KeyError):
            xgboost_feature_engineering(ohlcv_df, features=["Ticker", "nonexistent"])


class TestMlpFeatureEngineering:
    def test_returns_dataframe(self, ohlcv_df):
        custom = ["Ticker", "Close", "Volume"]
        result = mlp_feature_engineering(ohlcv_df, features=custom)
        assert isinstance(result, pd.DataFrame)

    def test_selects_specified_columns(self, ohlcv_df):
        custom = ["Ticker", "Close", "Volume"]
        result = mlp_feature_engineering(ohlcv_df, features=custom)
        assert list(result.columns) == custom

    def test_missing_column_raises(self, ohlcv_df):
        with pytest.raises(KeyError):
            mlp_feature_engineering(ohlcv_df, features=["Ticker", "nonexistent"])


def test_default_features_selection(ohlcv_df):
    df = ohlcv_df.copy()
    for col in ["log_return_lag_1", "log_return_lag_5", "log_return_lag_10", "log_return_lag_20",
                "log_distance_close_vs_SMA_5", "log_distance_close_vs_SMA_10", "log_distance_close_vs_SMA_20",
                "log_return_volatility_5", "log_return_volatility_10", "log_return_volatility_20",
                "bollinger_upper_distance_20", "bollinger_lower_distance_20", "bollinger_bandwidth_20",
                "rsi_14", "range_percentage", "body_percentage", "upper_shadow_pct", "lower_shadow_pct",
                "volume_pct_change_1", "relative_volume_5", "relative_volume_20", "drawdown_20",
                "sharpe_20", "day_of_week", "month", "market_movement",
                "log_return_lag_2", "log_return_lag_3", "log_distance_close_vs_SMA_50",
                "ema5_minus_ema20", "ema12_minus_ema26", "bollinger_upper_distance_10",
                "bollinger_lower_distance_10", "bollinger_bandwidth_10", "rsi_7", "rsi_21",
                "daily_range", "candle_body", "upper_shadow", "lower_shadow", "volume_sma_20",
                "return_x_volume", "drawdown_10", "drawdown_50", "rolling_window_mdd_10",
                "rolling_window_mdd_20", "rolling_window_mdd_50", "sharpe_10", "momentum_1", "momentum_10"]:
        df[col] = 1.0

    lr_res = lr_feature_engineering(df)
    xgb_res = xgboost_feature_engineering(df)
    mlp_res = mlp_feature_engineering(df)
    assert len(lr_res.columns) == 27
    assert len(xgb_res.columns) == 51
    assert len(mlp_res.columns) == 30


def test_merge_dataframes():
    stock = pd.DataFrame({"Date": ["2024-01-01"], "f1": [1], "Repaired?": [True]})
    spy = pd.DataFrame({"Date": ["2024-01-01"], "spy1": [2]})
    vix = pd.DataFrame({"Date": ["2024-01-01"], "vix1": [3]})

    from financial_forecasting_platform.pipelines.feature_engineering.nodes import merge_dataframes
    merged = merge_dataframes(stock, spy, vix)
    assert "Repaired?" not in merged.columns
    assert "f1" in merged.columns and "spy1" in merged.columns and "vix1" in merged.columns


class TestFeatureEngineeringPipeline:
    def test_returns_pipeline(self):
        result = create_pipeline()
        assert isinstance(result, Pipeline)

    def test_pipeline_has_seven_nodes(self):
        pipeline = create_pipeline()
        assert len(pipeline.nodes) == 7

    def test_pipeline_node_names(self):
        pipeline = create_pipeline()
        node_names = {n.name for n in pipeline.nodes}
        expected_names = {
            "create_stock_features_node",
            "create_spy_features_node",
            "create_vix_features_node",
            "merge_data_node",
            "lr_feature_engineering_node",
            "xgboost_feature_engineering_node",
            "mlp_feature_engineering_node",
        }
        assert node_names == expected_names

    def test_pipeline_inputs(self):
        pipeline = create_pipeline()
        input_datasets = set()
        for n in pipeline.nodes:
            input_datasets.update(n.inputs)
        assert "validated_raw_data" in input_datasets
        assert "validated_spy_data" in input_datasets
        assert "validated_vix_data" in input_datasets

    def test_pipeline_outputs(self):
        pipeline = create_pipeline()
        output_datasets = set()
        for n in pipeline.nodes:
            output_datasets.update(n.outputs)
        assert "stock_features" in output_datasets
        assert "spy_features" in output_datasets
        assert "vix_features" in output_datasets
        assert "all_features" in output_datasets
        assert "lr_features" in output_datasets
        assert "xgboost_features" in output_datasets
        assert "mlp_features" in output_datasets
