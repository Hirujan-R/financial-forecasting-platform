import pandas as pd
from sklearn.ensemble import IsolationForest

def outlier_detection(df: pd.DataFrame, 
                      outlier_feature_selection: list | None = None) -> pd.DataFrame:
    if outlier_feature_selection is None:
        outlier_feature_selection = [
            'upper_shadow_pct', 'lower_shadow_pct', 'drawdown_20', 
            'range_percentage', 'relative_volume_5', 'log_return_lag_1',
            'bollinger_lower_distance_20', 'log_return_volatility_5'
        ]
    return_df = df.copy()
    iso_forest = IsolationForest(contamination=0.015, random_state=42)
    return_df['is_outlier'] = (iso_forest.fit_predict(return_df[outlier_feature_selection]) == -1).astype(int)
    return return_df

def clip_outliers(df: pd.DataFrame, 
                  clip_columns: list | None = None) -> pd.DataFrame:
    if clip_columns is None:
        clip_columns = [
            'upper_shadow_pct', 'lower_shadow_pct', 'drawdown_20', 
            'range_percentage', 'relative_volume_5', 'log_return_lag_1',
            'bollinger_lower_distance_20', 'bollinger_upper_distance_20', 'bollinger_bandwidth_20', 'log_return_volatility_5', 'volume_pct_change_1', 
            'relative_volume_20', 'body_percentage', "gk_variance_mean_5_lag_1", "gk_variance_mean_20_lag_1", "parkinson_volatility_5_lag_1", "parkinson_volatility_20_lag_1", 
            "rs_volatility_5_lag_1", "rs_volatility_20_lag_1", "yz_volatility_5_lag_1", "yz_volatility_20_lag_1", "yz_volatility_60_lag_1", "yz_volatility_ratio_5_20", "yz_volatility_ratio_20_60",
            'spy_lag_return_1', 'spy_lag_return_5', 'spy_lag_return_20', 'spy_volatility_20', 'spy_drawdown_252', 'spy_sma_ratio_50', 'spy_sma_ratio_200', 'spy_return_zscore_252', 'vix_level',
            'vix_return_lag_1', 'vix_return_lag_5', 'vix_return_lag_20', 'vix_SMA_10', 'vix_SMA_20', 'vix_SMA_60',
        ]
    return_df = df.copy()
    for column in clip_columns:
        lower_quantile = return_df[column].quantile(0.01)
        higher_quantile = return_df[column].quantile(0.99)
        return_df[column] = return_df[column].clip(lower=lower_quantile, upper=higher_quantile)
    return return_df
