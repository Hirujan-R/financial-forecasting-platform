import pandas as pd
from financial_forecasting_platform.features.engineering import create_lag_return_feature, \
    create_log_return_lag_feature, create_simple_moving_average_feature, create_ema_feature, \
    create_distance_from_sma_feature, create_rsi_feature, create_momentum_feature, \
    create_simple_return_volatility_feature, create_log_return_volatility_feature, \
    create_bollinger_bands_features, create_daily_range_feature, create_range_percentage_feature, \
    create_candle_body_feature, create_upper_shadow_feature, create_lower_shadow_feature, \
    create_upper_shadow_percentage_feature, create_lower_shadow_percentage_feature, \
    create_volume_pct_change_feature, create_relative_volume_feature, \
    create_volume_sma_feature, create_return_x_volume_feature, create_drawdown_feature, \
    create_rolling_sharpe_ratio_feature, create_date_features, create_market_movement_target

def feature_engineering(df: pd.DataFrame):

    return_df = df.copy()
    return_df.sort_index(inplace=True)

    # Returns
    return_df = create_lag_return_feature(return_df, lag=1)
    return_df = create_lag_return_feature(return_df, lag=5)
    return_df = create_lag_return_feature(return_df, lag=10)
    return_df = create_lag_return_feature(return_df, lag=20)
    return_df = create_log_return_lag_feature(return_df, lag=1)

    # Trends
    return_df = create_simple_moving_average_feature(return_df, window_size = 5)
    return_df = create_simple_moving_average_feature(return_df, window_size = 20)
    return_df = create_simple_moving_average_feature(return_df, window_size = 50)
    return_df = create_ema_feature(return_df, window_size = 12)
    return_df = create_ema_feature(return_df, window_size = 26)
    return_df = create_distance_from_sma_feature(return_df, window_size = 20)
    # MACD

    # Momentum
    return_df = create_rsi_feature(return_df, window_size=14)
    return_df = create_momentum_feature(return_df, lag=5)
    return_df = create_momentum_feature(return_df, lag=20)
    return_df = create_momentum_feature(return_df, lag=60)


    # Volatility
    return_df = create_simple_return_volatility_feature(return_df, window_size=5)
    return_df = create_simple_return_volatility_feature(return_df, window_size=20)
    return_df = create_simple_return_volatility_feature(return_df, window_size=60)
    return_df = create_log_return_volatility_feature(return_df, window_size=5)
    return_df = create_log_return_volatility_feature(return_df, window_size=20)
    return_df = create_log_return_volatility_feature(return_df, window_size=60)
    return_df = create_bollinger_bands_features(return_df, window_size = 30)

    # Price Action
    return_df = create_daily_range_feature(return_df)
    return_df = create_range_percentage_feature(return_df)
    return_df = create_candle_body_feature(return_df)
    return_df = create_upper_shadow_feature(return_df)
    return_df = create_lower_shadow_feature(return_df)
    return_df = create_upper_shadow_percentage_feature(return_df)
    return_df = create_lower_shadow_percentage_feature(return_df)

    # Volume
    return_df = create_volume_pct_change_feature(return_df, lag=1)
    return_df = create_relative_volume_feature(return_df, window_size=30)
    return_df = create_volume_sma_feature(return_df, window_size=20)
    return_df = create_return_x_volume_feature(return_df)

    # Risk
    return_df = create_drawdown_feature(return_df, window_size=30)
    return_df = create_rolling_sharpe_ratio_feature(return_df, window_size=30)

    # Date
    return_df = create_date_features(return_df)
    


    return return_df

def target_variable_engineering(df: pd.DataFrame):
    return create_market_movement_target(df)