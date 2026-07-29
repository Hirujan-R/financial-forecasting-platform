import pandas as pd
from financial_forecasting_platform.features.engineering import create_market_movement_target
import financial_forecasting_platform.features.engineering as fe

FEATURE_FUNCTION_REGISTRY = {
    name: getattr(fe, name)
    for name in dir(fe)
    if name.startswith("create_")
    and name not in ("create_all_features")
}

def create_all_features(df: pd.DataFrame, feature_config: list, 
                        columns_to_drop: list | None = None, 
                        date_column: str = "Date") -> pd.DataFrame:
    return_df = df.copy()

    for feature in feature_config:
        function_name = feature["function"]
        kwargs = feature.get("kwargs", {})
        try:
            func = FEATURE_FUNCTION_REGISTRY[function_name]
        except KeyError:
            raise ValueError(
                f"Feature engineering function '{function_name}' is not "
                f"registered in FEATURE_FUNCTION_REGISTRY. Registered "
                f"functions: {sorted(FEATURE_FUNCTION_REGISTRY)}"
            )
        return_df = func(df=return_df, **kwargs)
    
    if columns_to_drop is None:
        columns_to_drop = ["Open","High","Low","Close","Volume"]
    return_df = return_df.drop(columns=columns_to_drop)
    return_df.set_index(date_column, inplace=True)

    return return_df


def lr_feature_engineering(df: pd.DataFrame, features: list | None = None) -> pd.DataFrame:
    if features is None:
        features = ["Ticker", "log_return_lag_1", "log_return_lag_5", "log_return_lag_10",
                    "log_return_lag_20", "log_distance_close_vs_SMA_5", 
                    "log_distance_close_vs_SMA_10", "log_distance_close_vs_SMA_20", 
                    "log_return_volatility_5", "log_return_volatility_10", 
                    "log_return_volatility_20", "bollinger_upper_distance_20", 
                    "bollinger_lower_distance_20", "bollinger_bandwidth_20", 
                    "rsi_14", "range_percentage", "body_percentage", 
                    "upper_shadow_pct", "lower_shadow_pct", "volume_pct_change_1", 
                    "relative_volume_5", "relative_volume_20", "drawdown_20", 
                    "sharpe_20", "day_of_week", "month", "market_movement"]
    return df[features]

def xgboost_feature_engineering(df: pd.DataFrame, features: list | None = None) -> pd.DataFrame:
    if features is None:
        features = ["Ticker", "log_return_lag_1", "log_return_lag_2", "log_return_lag_3", 
                    "log_return_lag_5", "log_return_lag_10", "log_return_lag_20", 
                    "log_distance_close_vs_SMA_5", "log_distance_close_vs_SMA_10", 
                    "log_distance_close_vs_SMA_20", "log_distance_close_vs_SMA_50",
                    "ema5_minus_ema20", "ema12_minus_ema26",
                    "log_return_volatility_5", "log_return_volatility_10", 
                    "log_return_volatility_20", "bollinger_upper_distance_10", 
                    "bollinger_lower_distance_10", "bollinger_bandwidth_10",
                    "bollinger_upper_distance_20", "bollinger_lower_distance_20", 
                    "bollinger_bandwidth_20", "rsi_7", "rsi_14", "rsi_21",
                    "daily_range", "range_percentage", 
                    "candle_body", "upper_shadow", "lower_shadow", "body_percentage", 
                    "upper_shadow_pct", "lower_shadow_pct", "volume_pct_change_1", 
                    "relative_volume_5", "relative_volume_20", "volume_sma_20", 
                    "return_x_volume", "drawdown_10", "drawdown_20", "drawdown_50",
                    "rolling_window_mdd_10", "rolling_window_mdd_20", 
                    "rolling_window_mdd_50", "sharpe_10", "sharpe_20", 
                    "momentum_1", "momentum_10", "day_of_week", "month", "market_movement"]
    return df[features]

def mlp_feature_engineering(df: pd.DataFrame, features: list | None = None) -> pd.DataFrame:
    if features is None:
        features = ["Ticker", "log_return_lag_1", "log_return_lag_3", "log_return_lag_5", "log_return_lag_10",
                    "log_return_lag_20", "log_distance_close_vs_SMA_5", 
                    "log_distance_close_vs_SMA_10", "log_distance_close_vs_SMA_20", 
                    "log_return_volatility_5", "log_return_volatility_10", 
                    "log_return_volatility_20", "bollinger_upper_distance_20", 
                    "bollinger_lower_distance_20", "bollinger_bandwidth_20", 
                    "rsi_7", "rsi_14", "range_percentage", "body_percentage", 
                    "upper_shadow_pct", "lower_shadow_pct", "volume_pct_change_1", 
                    "relative_volume_5", "relative_volume_20", "drawdown_20", 
                    "rolling_window_mdd_20", "sharpe_20", "day_of_week", "month", "market_movement"]
    return df[features]

def merge_dataframes(stock_df: pd.DataFrame, spy_df: pd.DataFrame, vix_df: pd.DataFrame) -> pd.DataFrame:
    merged_data = stock_df.merge(spy_df, on="Date", how="left").merge(vix_df, on="Date", how="left")
    if 'Repaired?' in merged_data.columns:
        merged_data = merged_data.drop(columns=['Repaired?'])
    return merged_data