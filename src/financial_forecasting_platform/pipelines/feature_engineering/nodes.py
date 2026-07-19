import pandas as pd
from financial_forecasting_platform.features.engineering import create_log_return_lag_feature, \
    create_ema_crossover_feature, create_log_distance_from_sma_feature, create_rsi_feature, \
    create_momentum_feature, create_log_return_volatility_feature, \
    create_bollinger_bands_features, create_daily_range_feature, create_range_percentage_feature, \
    create_candle_body_feature, create_upper_shadow_feature, create_lower_shadow_feature, \
    create_body_percentage_feature, create_upper_shadow_percentage_feature, create_lower_shadow_percentage_feature, \
    create_volume_pct_change_feature, create_relative_volume_feature, \
    create_volume_sma_feature, create_return_x_volume_feature, create_drawdown_feature, \
    create_rolling_window_mdd_feature, create_rolling_sharpe_ratio_feature, \
    create_date_features, create_market_movement_target


def lr_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    return_df = df.copy()

    # Log Return
    return_df = create_log_return_lag_feature(df=return_df, lag=1)
    return_df = create_log_return_lag_feature(df=return_df, lag=5)
    return_df = create_log_return_lag_feature(df=return_df, lag=10)
    return_df = create_log_return_lag_feature(df=return_df, lag=20)

    # Trend Positioning
    return_df = create_log_distance_from_sma_feature(df=return_df, window_size=5)
    return_df = create_log_distance_from_sma_feature(df=return_df, window_size=10)
    return_df = create_log_distance_from_sma_feature(df=return_df, window_size=20)

    # Volatility
    return_df = create_log_return_volatility_feature(df=return_df, window_size=5)
    return_df = create_log_return_volatility_feature(df=return_df, window_size=10)
    return_df = create_log_return_volatility_feature(df=return_df, window_size=20)

    # Volatility regime
    return_df = create_bollinger_bands_features(df=return_df, window_size=20)
    return_df = create_rsi_feature(df=return_df, window_size=14)

    # Candle structure
    return_df = create_range_percentage_feature(df=return_df)
    return_df = create_body_percentage_feature(df=return_df)
    return_df = create_upper_shadow_percentage_feature(df=return_df)
    return_df = create_lower_shadow_percentage_feature(df=return_df)

    # Volume
    return_df = create_volume_pct_change_feature(df=return_df, lag=1)
    return_df = create_relative_volume_feature(df=return_df, window_size=5)
    return_df = create_relative_volume_feature(df=return_df, window_size=20)

    # Risk
    return_df = create_drawdown_feature(df=return_df, window_size=20)
    return_df = create_rolling_sharpe_ratio_feature(df=return_df, window_size=20)

    # Date
    return_df = create_date_features(df=return_df)

    return_df = return_df.drop(columns=["Open","High","Low","Close","Volume"])
    return_df.set_index("Date", inplace=True)

    return return_df

def xgboost_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    return_df = df.copy()

    # Log Return
    return_df = create_log_return_lag_feature(df=return_df, lag=1)
    return_df = create_log_return_lag_feature(df=return_df, lag=2)
    return_df = create_log_return_lag_feature(df=return_df, lag=3)
    return_df = create_log_return_lag_feature(df=return_df, lag=5)
    return_df = create_log_return_lag_feature(df=return_df, lag=10)
    return_df = create_log_return_lag_feature(df=return_df, lag=20)
    return_df = create_log_distance_from_sma_feature(df=return_df, window_size=50)

    # Trend Positioning
    return_df = create_log_distance_from_sma_feature(df=return_df, window_size=5)
    return_df = create_log_distance_from_sma_feature(df=return_df, window_size=10)
    return_df = create_log_distance_from_sma_feature(df=return_df, window_size=20)

    # Crossover
    return_df = create_ema_crossover_feature(df=return_df, short_span=5, long_span=20)
    return_df = create_ema_crossover_feature(df=return_df, short_span=12, long_span=26)

    # Volatility
    return_df = create_log_return_volatility_feature(df=return_df, window_size=5)
    return_df = create_log_return_volatility_feature(df=return_df, window_size=10)
    return_df = create_log_return_volatility_feature(df=return_df, window_size=20)

    # Volatility regime
    return_df = create_bollinger_bands_features(df=return_df, window_size=10)
    return_df = create_bollinger_bands_features(df=return_df, window_size=20)
    return_df = create_rsi_feature(df=return_df, window_size=7)
    return_df = create_rsi_feature(df=return_df, window_size=14)
    return_df = create_rsi_feature(df=return_df, window_size=21)

    # Candle structure
    return_df = create_daily_range_feature(df=return_df)
    return_df = create_range_percentage_feature(df=return_df)
    return_df = create_candle_body_feature(df=return_df)
    return_df = create_upper_shadow_feature(df=return_df)
    return_df = create_lower_shadow_feature(df=return_df)
    return_df = create_body_percentage_feature(df=return_df)
    return_df = create_upper_shadow_percentage_feature(df=return_df)
    return_df = create_lower_shadow_percentage_feature(df=return_df)
    
    # Volume
    return_df = create_volume_pct_change_feature(df=return_df, lag=1)
    return_df = create_relative_volume_feature(df=return_df, window_size=5)
    return_df = create_relative_volume_feature(df=return_df, window_size=20)
    return_df = create_volume_sma_feature(df=return_df, window_size=20)
    return_df = create_return_x_volume_feature(df=return_df)

    # Risk
    return_df = create_drawdown_feature(df=return_df, window_size=10)
    return_df = create_drawdown_feature(df=return_df, window_size=20)
    return_df = create_drawdown_feature(df=return_df, window_size=50)
    return_df = create_rolling_window_mdd_feature(df=return_df, window_size=10)
    return_df = create_rolling_window_mdd_feature(df=return_df, window_size=20)
    return_df = create_rolling_window_mdd_feature(df=return_df, window_size=50)
    return_df = create_rolling_sharpe_ratio_feature(df=return_df, window_size=10)
    return_df = create_rolling_sharpe_ratio_feature(df=return_df, window_size=20)

    # Momentum
    return_df = create_momentum_feature(df=return_df, lag=1)
    return_df = create_momentum_feature(df=return_df, lag=10)

    # Date
    return_df = create_date_features(df=return_df)

    return_df = return_df.drop(columns=["Open","High","Low","Close","Volume"])
    return_df.set_index("Date", inplace=True)

    return return_df

def mlp_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    return_df = df.copy()

    # Log Return
    return_df = create_log_return_lag_feature(df=return_df, lag=1)
    return_df = create_log_return_lag_feature(df=return_df, lag=3)
    return_df = create_log_return_lag_feature(df=return_df, lag=5)
    return_df = create_log_return_lag_feature(df=return_df, lag=10)
    return_df = create_log_return_lag_feature(df=return_df, lag=20)

    # Trend Positioning
    return_df = create_log_distance_from_sma_feature(df=return_df, window_size=5)
    return_df = create_log_distance_from_sma_feature(df=return_df, window_size=10)
    return_df = create_log_distance_from_sma_feature(df=return_df, window_size=20)

    # Volatility
    return_df = create_log_return_volatility_feature(df=return_df, window_size=5)
    return_df = create_log_return_volatility_feature(df=return_df, window_size=10)
    return_df = create_log_return_volatility_feature(df=return_df, window_size=20)

    # Volatility regime
    return_df = create_bollinger_bands_features(df=return_df, window_size=20)
    return_df = create_rsi_feature(df=return_df, window_size=7)
    return_df = create_rsi_feature(df=return_df, window_size=14)

    # Candle structure
    return_df = create_range_percentage_feature(df=return_df)
    return_df = create_body_percentage_feature(df=return_df)
    return_df = create_upper_shadow_percentage_feature(df=return_df)
    return_df = create_lower_shadow_percentage_feature(df=return_df)

    # Volume
    return_df = create_volume_pct_change_feature(df=return_df, lag=1)
    return_df = create_relative_volume_feature(df=return_df, window_size=5)
    return_df = create_relative_volume_feature(df=return_df, window_size=20)

    # Risk
    return_df = create_drawdown_feature(df=return_df, window_size=20)
    return_df = create_rolling_window_mdd_feature(df=return_df, window_size=20)
    return_df = create_rolling_sharpe_ratio_feature(df=return_df, window_size=20)

    # Date
    return_df = create_date_features(df=return_df)

    return_df = return_df.drop(columns=["Open","High","Low","Close","Volume"])
    return_df.set_index("Date", inplace=True)

    return return_df

def target_variable_engineering(df: pd.DataFrame):
    return create_market_movement_target(df)