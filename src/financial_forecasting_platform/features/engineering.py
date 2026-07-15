import pandas as pd
import numpy as np
import warnings

def create_market_movement_target(df: pd.DataFrame) -> pd.Series:
    return_df = df.copy()
    return_df.sort_values("Date", inplace=True)

    market_difference = return_df.groupby("Ticker")["Close"].diff(-1)
    market_movement = np.where(
        market_difference.isna(), np.nan, (market_difference < 0).astype(float)
    )

    return pd.Series(
    market_movement,
    index=return_df.index,
    name="market_movement"
    )

def create_close_lag_feature(df: pd.DataFrame, number_of_days: int = 1) -> pd.DataFrame:
    if number_of_days < 1:
        raise ValueError("number_of_days must be a positive integer.")
        
    return_df = df.copy()
    return_df.sort_values("Date", inplace=True)

    col_name = f"close_lag_{number_of_days}"
    return_df[col_name] = return_df.groupby("Ticker")["Close"].shift(
        number_of_days
    )

    return return_df

def create_momentum_feature(df: pd.DataFrame, lag: int = 1) -> pd.DataFrame:
    if lag < 1:
        raise ValueError("lag must be a positive integer.")
        
    return_df = df.copy()
    return_df.sort_values("Date", inplace=True)

    column_name = f"momentum_{lag}"
    return_df[column_name] = return_df.groupby("Ticker")["Close"].diff(lag)
    return return_df

def create_lag_return_feature(df: pd.DataFrame, lag: int = 1) -> pd.DataFrame:
    if lag < 1:
        raise ValueError("lag must be a positive integer.")

    return_df = df.copy()
    return_df.sort_values("Date", inplace=True)

    lag_close = return_df.groupby("Ticker")["Close"].shift(lag).replace(0.0, np.nan)
    return_df[f"return_lag_{lag}"] = (return_df["Close"] / lag_close) - 1

    return return_df

def create_log_return_lag_feature(df: pd.DataFrame, lag: int = 1):
    if lag < 1:
        raise ValueError("lag must be a positive integer.")
    return_df = df.copy()
    return_df.sort_values("Date", inplace=True)
    lag_close = return_df.groupby("Ticker")["Close"].shift(lag).replace(0.0, np.nan)
    return_df[f"log_return_lag_{lag}"] = np.log((return_df["Close"] / lag_close)) 
    return return_df

def create_simple_moving_average_feature(df: pd.DataFrame, window_size: int = 2) -> pd.DataFrame:
    if window_size < 1:
        raise ValueError("window_size must be a positive integer.")
    if window_size == 1:
        warnings.warn("Setting window_size = 1 is redundant. Use a value larger than 1.")
        
    return_df = df.copy()
    return_df.sort_values("Date", inplace=True)
    
    column_name = f"SMA_{window_size}"
    return_df[column_name] = return_df.groupby("Ticker")["Close"].transform(lambda x: x.rolling(window=window_size, min_periods=window_size).mean())
    return return_df

def create_distance_from_sma_feature(df: pd.DataFrame, window_size: int = 2) -> pd.DataFrame:
    if window_size < 1:
        raise ValueError("window_size must be a positive integer.")
    if window_size == 1:
        warnings.warn("Setting window_size = 1 is redundant. Use a value larger than 1.")
    
    return_df = df.copy()
    return_df.sort_values("Date", inplace=True)

    column_name = f"close_vs_SMA_{window_size}"
    sma = return_df.groupby("Ticker")["Close"].transform(lambda x: x.rolling(window=window_size, min_periods=window_size).mean()).replace(0, np.nan)
    return_df[column_name] = (return_df["Close"] - sma) / sma
    return return_df

def create_ema_feature(df: pd.DataFrame, window_size: int = 2) -> pd.DataFrame:
    if window_size < 1:
        raise ValueError("window_size must be a positive integer.")
    if window_size == 1:
        warnings.warn("Setting window_size = 1 is redundant. Use a value larger than 1.")
        
    return_df = df.copy()
    return_df.sort_values("Date", inplace=True)
    
    column_name = f"ema_{window_size}"
    return_df[column_name] = return_df.groupby("Ticker")["Close"].transform(lambda x: x.ewm(span = window_size, adjust = False, min_periods = window_size).mean())
    return return_df

def create_ema_crossover_feature(df: pd.DataFrame, short_span: int, long_span: int) -> pd.DataFrame:
    if short_span == long_span:
        raise ValueError("Short span and long span must be different.")
    if short_span < 1 or long_span < 1:
        raise ValueError("Span sizes must be positive integers.")
    
    return_df = df.copy()
    return_df.sort_values("Date", inplace=True)
    
    column_name = f"ema{short_span}_minus_ema{long_span}"
    ema_short_span = return_df.groupby("Ticker")["Close"].transform(lambda x: x.ewm(span = short_span, adjust = False, min_periods = short_span).mean())
    ema_long_span = return_df.groupby("Ticker")["Close"].transform(lambda x: x.ewm(span = long_span, adjust = False, min_periods = long_span).mean())
    return_df[column_name] = ema_short_span - ema_long_span
    return return_df

def create_simple_return_volatility_feature(df: pd.DataFrame, window_size: int = 2) -> pd.DataFrame:
    if window_size <= 1:
        raise ValueError("window_size must be an integer greater than 1.")
        
    return_df = df.copy()
    return_df.sort_values("Date", inplace=True)
    
    yesterday_close = return_df.groupby("Ticker")["Close"].shift(1).replace(0.0, np.nan)
    daily_returns = (return_df["Close"] - yesterday_close) / yesterday_close
    column_name = f"simple_return_volatility_{window_size}"
    return_df[column_name] = daily_returns.groupby(return_df["Ticker"]).transform(lambda x: x.rolling(window = window_size).std())
    return return_df

def create_log_return_volatility_feature(df: pd.DataFrame, window_size: int = 2) -> pd.DataFrame:
    if window_size <= 1:
        raise ValueError("window_size must be an integer greater than 1.")
        
    return_df = df.copy()
    return_df.sort_values("Date", inplace=True)
    
    yesterday_close = return_df.groupby("Ticker")["Close"].shift(1).replace(0.0, np.nan)
    daily_returns = np.log(return_df["Close"] / yesterday_close)
    column_name = f"log_return_volatility_{window_size}"
    return_df[column_name] = daily_returns.groupby(return_df["Ticker"]).transform(lambda x: x.rolling(window = window_size).std())
    return return_df

def create_bollinger_bands_features(df: pd.DataFrame, window_size: int = 20) -> pd.DataFrame:
    """Bollinger bands are used to measure market volatility and whether a price is unusually high or low.
       Middle band is typically the 20-day SMA. The upper band is (SMA + 2σ) and the lower band is
       SMA -2σ) standard deviations. Touching the upper band suggests the asset is overbought and touching the lower
       band suggests the asset is oversold. When the bands constrict this suggests the volatility is dropping
       (The Squeeze) and there is going to be a sharp breakout (sudden increase) as volatility is cyclical.

       The raw bands are highly correlated with price thus don't carry much useful information. However they can be 
       normalised by calculating distance and bandwidth.
       
       This function engineers the features: 
           upper band distance (upper band - Close) / Close, 
           lower band distance (Close - lower band) / Close, 
           bandwidth (upper band - lower band) / middle band."""

    if window_size <= 1:
        raise ValueError("window_size must be an integer greater than 1.")
        
    return_df = df.copy()
    return_df.sort_values("Date", inplace=True)
    
    sma = return_df.groupby("Ticker")["Close"].transform(lambda x: x.rolling(window=window_size, min_periods=window_size).mean())
    standard_deviation = return_df.groupby("Ticker")["Close"].transform(lambda x: x.rolling(window=window_size, min_periods=window_size).std())
    upper_band = sma + 2*standard_deviation
    lower_band = sma - 2*standard_deviation
    close = return_df["Close"].replace(0.0, np.nan)
    
    
    return_df[f"bollinger_upper_distance_{window_size}"] = (upper_band - return_df["Close"]) / close
    return_df[f"bollinger_lower_distance_{window_size}"] = (return_df["Close"] - lower_band) / close
    return_df[f"bollinger_bandwidth_{window_size}"] = (standard_deviation * 4) / sma.replace(0.0, np.nan)

    return return_df

def create_rsi_feature(df: pd.DataFrame, window_size: int = 14) -> pd.DataFrame:
    """Relative strength index (RSI) is a momentum oscillator to measure the speed 
       and change of price movements. RSI oscillates between 0 and 100 and is used
       to identify overbought (above 70) and oversold (under 30) conditions in a market.

       RSI = 100 - (100 / (1 + RS))
       RS = average gain / average loss

       Averages are calculated using Wilder's smoothing such that:
           average_gain_t = ((average_gain_t-1 * (window_size-1)) + gain_t) / window_size
           average_loss_t = ((average_loss_t-1 * (window_size-1)) + loss_t) / window_size
           
       if price_change > 0: gain_t = price_change else gain_t = 0
       if price_change < 0: loss_t = -price_change else loss_t = 0

       This function engineers the RSI value."""

    if window_size <= 1:
        raise ValueError("window_size must be an integer greater than 1.")
        
    return_df = df.copy()
    return_df.sort_values("Date", inplace=True)
    
    price_difference = return_df.groupby("Ticker")["Close"].diff()
    gain = price_difference.clip(lower = 0)
    loss = -price_difference.clip(upper = 0)

    avg_gain = gain.groupby(return_df["Ticker"]).transform(lambda x: x.ewm(alpha = 1/window_size, min_periods = window_size, adjust=False, ignore_na=True).mean())
    avg_loss = loss.groupby(return_df["Ticker"]).transform(lambda x: x.ewm(alpha = 1/window_size, min_periods = window_size, adjust=False, ignore_na=True).mean())

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi[(avg_loss == 0) & (avg_gain > 0)] = 100
    rsi[(avg_gain == 0) & (avg_loss == 0)] = 50
    return_df[f"rsi_{window_size}"] = rsi
    return return_df
    
def create_daily_range_feature(df: pd.DataFrame) -> pd.DataFrame:
    return_df = df.copy()
    return_df.sort_values("Date", inplace=True)
    return_df["daily_range"] = return_df["High"] - return_df["Low"]
    return return_df

def create_range_percentage_feature(df: pd.DataFrame) -> pd.DataFrame:
    return_df = df.copy()
    return_df.sort_values("Date", inplace=True)
    return_df["range_percentage"] = (return_df["High"] - return_df["Low"]) / return_df["Close"]
    return return_df

def create_candle_body_feature(df: pd.DataFrame) -> pd.DataFrame:
    return_df = df.copy()
    return_df.sort_values("Date", inplace=True)
    return_df["candle_body"] = return_df["Close"] - return_df["Open"]
    return return_df

def create_body_percentage_feature(df: pd.DataFrame) -> pd.DataFrame:
    return_df = df.copy()
    return_df.sort_values("Date", inplace=True)
    return_df["body_percentage"] = (return_df["Close"] - return_df["Open"]) / return_df["Open"]
    return return_df

def create_upper_shadow_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates upper shadow (upper wick) feature.

    Upper shadow = High - max(Open, Close)
    """

    return_df = df.copy()
    return_df.sort_values("Date", inplace=True)

    return_df["upper_shadow"] = return_df["High"] - return_df[["Open", "Close"]].max(axis=1)

    return return_df

def create_lower_shadow_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates lower shadow (lower wick) feature.

    Lower shadow = min(Open, Close) - Low
    """

    return_df = df.copy()
    return_df.sort_values("Date", inplace=True)

    return_df["lower_shadow"] = return_df[["Open", "Close"]].min(axis=1) - return_df["Low"]

    return return_df

def create_upper_shadow_percentage_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates upper shadow (upper wick) percentage feature.

    Upper shadow percentage = (High - max(Open, Close)) / Close
    """

    return_df = df.copy()
    return_df.sort_values("Date", inplace=True)

    upper_shadow = return_df["High"] - return_df[["Open", "Close"]].max(axis=1)
    return_df["upper_shadow_pct"] = upper_shadow / return_df["Close"].replace(0, np.nan)

    return return_df

def create_lower_shadow_percentage_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates lower shadow (lower wick) percentage feature.

    Lower shadow percentage = (min(Open, Close) - Low) / Close
    """

    return_df = df.copy()
    return_df.sort_values("Date", inplace=True)

    lower_shadow = return_df[["Open", "Close"]].min(axis=1) - return_df["Low"]
    return_df["lower_shadow_pct"] = lower_shadow / return_df["Close"].replace(0, np.nan)

    return return_df

def create_volume_pct_change_feature(df: pd.DataFrame, lag: int = 1) -> pd.DataFrame:
    if lag < 1:
        raise ValueError("lag must be an integer greater than 1.")
    return_df = df.copy()
    return_df.sort_values("Date", inplace=True)
    lag_volume = return_df.groupby("Ticker")["Volume"].shift(lag)
    return_df["volume_pct_change"] = (return_df["Volume"] - lag_volume) / lag_volume
    return return_df

def create_volume_sma_feature(df: pd.DataFrame, window_size: int = 2) -> pd.DataFrame:
    if window_size < 1:
        raise ValueError("window_size must be an integer greater than 1.")
    if window_size == 1:
        warnings.warn("Setting window_size = 1 is redundant. Use a value larger than 1.")
        
    return_df = df.copy()
    return_df.sort_values("Date", inplace=True)

    column_name = f"volume_sma_{window_size}"
    return_df[column_name] = return_df.groupby("Ticker")["Volume"].transform(lambda x: x.rolling(window=window_size, min_periods=window_size).mean())
    return return_df

def create_relative_volume_feature(df: pd.DataFrame, window_size: int = 2) -> pd.DataFrame:
    """Relative volume = volume today / Average volume"""
    if window_size < 1:
        raise ValueError("window_size must be an integer greater than 1.")
    if window_size == 1:
        warnings.warn("Setting window_size = 1 is redundant. Use a value larger than 1.")
        
    return_df = df.copy()
    return_df.sort_values("Date", inplace=True)

    column_name = f"relative_volume_{window_size}"
    return_df[column_name] = return_df["Volume"] / return_df.groupby("Ticker")["Volume"].transform(lambda x: x.rolling(window=window_size, min_periods=window_size).mean())
    return return_df

def create_return_x_volume_feature(df: pd.DataFrame) -> pd.DataFrame:
    
    return_df = df.copy()
    return_df.sort_values("Date", inplace=True)

    price_return = return_df.groupby("Ticker")["Close"].diff()
    return_df["return_x_volume"] = return_df["Volume"] * price_return
    return return_df

def create_drawdown_feature(df: pd.DataFrame, window_size: int = 2) -> pd.DataFrame:
    if window_size <= 1:
        raise ValueError("window_size must be an integer greater than 1.")
    return_df = df.copy()
    return_df.sort_values("Date", inplace=True)
    peak = return_df.groupby("Ticker")["Close"].transform(lambda x: x.rolling(window=window_size, min_periods=window_size).max()).replace(0.0, np.nan)
    return_df[f"drawdown_{window_size}"]= (return_df["Close"] - peak) / peak
    return return_df

def create_rolling_window_mdd_feature(df: pd.DataFrame, window_size: int = 2) -> pd.DataFrame:
    if window_size <= 1:
        raise ValueError("window_size must be an integer greater than 1.")
    return_df = df.copy()
    return_df.sort_values("Date", inplace=True)
    peak = return_df.groupby("Ticker")["Close"].transform(lambda x: x.rolling(window=window_size, min_periods=window_size).max()).replace(0.0, np.nan)
    drawdown = ((return_df["Close"] - peak) / peak)
    return_df[f"rolling_window_mdd_{window_size}"] = drawdown.groupby(return_df["Ticker"]).transform(lambda x: x.rolling(window=window_size, min_periods=window_size).min())
    return return_df

def create_date_features(df: pd.DataFrame) -> pd.DataFrame:
    return_df = df.copy()
    return_df.sort_values("Date", inplace=True)
    return_df["day_of_week"] = return_df["Date"].transform(lambda x: x.strftime('%A'))
    return_df["month"] = return_df["Date"].transform(lambda x: x.strftime('%m'))
    return return_df

def create_rolling_sharpe_ratio_feature(df: pd.DataFrame, window_size: int = 2) -> pd.DataFrame:
    """"""
    if window_size <= 1:
        raise ValueError("window_size must be an integer greater than 1.")
    return_df = df.copy()
    return_df.sort_values("Date", inplace=True)

    yesterday_close = return_df.groupby("Ticker")["Close"].shift().replace(0, np.nan)
    daily_return = (return_df["Close"] - yesterday_close) / yesterday_close
    mean_return = daily_return.groupby(return_df["Ticker"]).transform(lambda x: x.rolling(window=window_size, min_periods=window_size).mean())
    std_return = daily_return.groupby(return_df["Ticker"]).transform(lambda x: x.rolling(window=window_size, min_periods=window_size).std()).replace(0.0, np.nan)
    return_df[f"sharpe_{window_size}"] = mean_return / std_return
    return return_df