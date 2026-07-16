import pandas as pd
import numpy as np
import warnings

def create_market_movement_target(df: pd.DataFrame) -> pd.Series:
    """Returns 1 if the Close price of next day is greater than Close price of 
       current day and 0 otherwise. Target variable of the model."""
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

def create_close_lag_feature(df: pd.DataFrame, lag: int = 1) -> pd.DataFrame:
    """Creates the lag close price feature. By default, lag is set to one."""
    if lag < 1:
        raise ValueError("lag must be a positive integer.")
        
    return_df = df.copy()
    return_df.sort_values("Date", inplace=True)

    col_name = f"close_lag_{lag}"
    return_df[col_name] = return_df.groupby("Ticker")["Close"].shift(
        lag
    )

    return return_df

def create_momentum_feature(df: pd.DataFrame, lag: int = 1) -> pd.DataFrame:
    """Creates momentum feature. Momentum shows speed of price changes / rate of price
       movement.
       momentum = close_price_t - close_price_t-lag.
       By default, lag is set to one."""
    if lag < 1:
        raise ValueError("lag must be a positive integer.")
        
    return_df = df.copy()
    return_df.sort_values("Date", inplace=True)

    column_name = f"momentum_{lag}"
    return_df[column_name] = return_df.groupby("Ticker")["Close"].diff(lag)
    return return_df

def create_lag_return_feature(df: pd.DataFrame, lag: int = 1) -> pd.DataFrame:
    """Creates simple return feature.
       Shows the profit or loss made on an investment over a specific period of time:
       return = (close_price_t / close_price_t_lag) - 1.
       return = (close_price_t - close_price_t_lag) / close_price_t_lag.
       By default, lag is set to one."""
    if lag < 1:
        raise ValueError("lag must be a positive integer.")

    return_df = df.copy()
    return_df.sort_values("Date", inplace=True)

    lag_close = return_df.groupby("Ticker")["Close"].shift(lag).replace(0.0, np.nan)
    return_df[f"return_lag_{lag}"] = (return_df["Close"] / lag_close) - 1

    return return_df

def create_log_return_lag_feature(df: pd.DataFrame, lag: int = 1):
    """Creates log return feature: 
       log_return = log(close_price_t / close_price_t_lag).
       By default, lag is set to one."""
    if lag < 1:
        raise ValueError("lag must be a positive integer.")
    return_df = df.copy()
    return_df.sort_values("Date", inplace=True)
    lag_close = return_df.groupby("Ticker")["Close"].shift(lag).replace(0.0, np.nan)
    return_df[f"log_return_lag_{lag}"] = np.log((return_df["Close"] / lag_close)) 
    return return_df

def create_simple_moving_average_feature(df: pd.DataFrame, window_size: int = 2) -> pd.DataFrame:
    """Creates SMA feature. SMA assigns equal weight to all points: 
       SMA = mean(close_price_t + close_price_t-1 + ... + close_price_t-window_size)
       By default, window_size is set to two."""
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
    """Creates distance from SMA feature. Shows how far current price is from average: 
       distance from SMA = (close_t - SMA(close_t,window_size) / SMA(close_t,window_size))
       By default, window_size is set to two."""
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
    """Creates EMA feature. EMA assigns exponentially larger weights to more recent points: 
       Example Multiplier = 2 / (num_periods + 1)
       EMA_t = (close_t - EMA_t-1) * Multiplier + EMA_t-1
       By default, window_size is set to two."""
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
    """Creates EMA crossover feature. Large EMA crossover shows potential shift 
       in market momentum and trend: 
       EMA_crossover_t = EMA(t,short_span) - EMA(t,long_span)"""
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
    """Creates return volocity feature. Shows speed at which money is returned: 
       volocity_t = std(daily_return_t, daily_return_t-1, ..., daily_return_t-window_size)"""
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
    """Creates log return volocity feature. Shows speed at which money is returned: 
       log_volocity_t = std(daily_log_return_t, ..., daily_log_return_t-window_size)"""
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
    """Creates daily range feature. Shows difference between High and 
       Low of an asset on a day: 
       daily_range = High - Low"""
    return_df = df.copy()
    return_df.sort_values("Date", inplace=True)
    return_df["daily_range"] = return_df["High"] - return_df["Low"]
    return return_df

def create_range_percentage_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Creates daily range feature. Shows difference between High and 
       Low of an asset on a day as a percentage: 
       daily_range_percentage = (High - Low) / Close"""
    return_df = df.copy()
    return_df.sort_values("Date", inplace=True)
    return_df["range_percentage"] = (return_df["High"] - return_df["Low"]) / return_df["Close"]
    return return_df

def create_candle_body_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Creates candle body feature. Shows price range between opening and closing price.
       If Close > Open -> price went up -> Bullish market.
       If Close < Open -> price went down -> Bearish market.
       Tall candle body -> strong buying pressure -> session dominated by buyers or sellers.
       Short candle body -> indecisive marker -> neither buyers or sellers dominated.
       candle_body = Close - Open"""
    return_df = df.copy()
    return_df.sort_values("Date", inplace=True)
    return_df["candle_body"] = return_df["Close"] - return_df["Open"]
    return return_df

def create_body_percentage_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Creates candle body percentage feature. Shows price range between opening 
       and closing price as percentage.
       Percentage shows the change between Close and Open relative to Open's value.
       If Close > Open -> price went up -> Bullish market.
       If Close < Open -> price went down -> Bearish market.
       Tall candle body -> strong buying pressure -> session dominated by buyers or sellers.
       Short candle body -> indecisive marker -> neither buyers or sellers dominated.
       candle_body_percentage = (Close - Open) / Open"""
    return_df = df.copy()
    return_df.sort_values("Date", inplace=True)
    return_df["body_percentage"] = (return_df["Close"] - return_df["Open"]) / return_df["Open"]
    return return_df

def create_upper_shadow_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Creates upper shadow (upper wick) feature. Measures failed attempt of buyers to
       keep prices elevated. 
       High upper shadow value -> Long upper shadow -> Aggressive selling pressure to push 
       price down.
       Low upper shadow value -> Short upper shadow -> Buyers dominated session as sellers 
       couldn't push down prices.

       Upper shadow = High - max(Open, Close)"""

    return_df = df.copy()
    return_df.sort_values("Date", inplace=True)

    return_df["upper_shadow"] = return_df["High"] - return_df[["Open", "Close"]].max(axis=1)

    return return_df

def create_lower_shadow_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Creates lower shadow (lower wick) feature. Measures failed attempt of sellers to
       keep prices down. 
       High lower shadow value -> Long lower shadow -> Aggressive buying pressure to push 
       price up.
       Low lower shadow value -> Short lower shadow -> Sellers dominated session as buyers 
       couldn't push up prices.

       Lower shadow = min(Open, Close) - Low"""

    return_df = df.copy()
    return_df.sort_values("Date", inplace=True)

    return_df["lower_shadow"] = return_df[["Open", "Close"]].min(axis=1) - return_df["Low"]

    return return_df

def create_upper_shadow_percentage_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Creates upper shadow percentage (upper wick) feature. 
       Measures failed attempt of buyers to keep prices elevated. 
       Percentage ensures value is relative.
       High upper shadow value -> Long upper shadow -> Aggressive selling pressure to push 
       price down.
       Low upper shadow value -> Short upper shadow -> Buyers dominated session as sellers 
       couldn't push down prices.

       upper_shadow_percentage = (High - max(Open, Close)) / Close"""

    return_df = df.copy()
    return_df.sort_values("Date", inplace=True)

    upper_shadow = return_df["High"] - return_df[["Open", "Close"]].max(axis=1)
    return_df["upper_shadow_pct"] = upper_shadow / return_df["Close"].replace(0, np.nan)

    return return_df

def create_lower_shadow_percentage_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Creates lower shadow (lower wick) percentage feature. Measures failed attempt 
       of sellers to keep prices down. 
       Percentage ensures value is relative.
       High lower shadow value -> Long lower shadow -> Aggressive buying pressure to push 
       price up.
       Low lower shadow value -> Short lower shadow -> Sellers dominated session as buyers 
       couldn't push up prices.

       lower_shadow_percentage = (min(Open, Close) - Low) / Close"""

    return_df = df.copy()
    return_df.sort_values("Date", inplace=True)

    lower_shadow = return_df[["Open", "Close"]].min(axis=1) - return_df["Low"]
    return_df["lower_shadow_pct"] = lower_shadow / return_df["Close"].replace(0, np.nan)

    return return_df

def create_volume_pct_change_feature(df: pd.DataFrame, lag: int = 1) -> pd.DataFrame:
    """Creates volume change percentage feature. Measures change in number of
       assets traded between period of time. 
       High volume percentage chage -> More assets traded during the time.
       Low volume percentage chage -> Less assets traded during the time.

       volume_pct_change = (Volume_t - Volume_t-lag) Volume_t-1"""
    if lag < 1:
        raise ValueError("lag must be an integer greater than 1.")
    return_df = df.copy()
    return_df.sort_values("Date", inplace=True)
    lag_volume = return_df.groupby("Ticker")["Volume"].shift(lag)
    return_df["volume_pct_change"] = (return_df["Volume"] - lag_volume) / lag_volume
    return return_df

def create_volume_sma_feature(df: pd.DataFrame, window_size: int = 2) -> pd.DataFrame:
    """Creates volume SMA feature. Measures average assets traded between period of time. 

       volume_SMA = mean(Volume_t + Volume_t-1 + ... + Volume_t-window_size)"""
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
    """Creates relative volume feature. Measures ratio of assets traded now compared 
       to average assets over period of time. 

       relative_volume = Volume_t - Volume_SMA_window_size"""
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
    """Creates return_x_volume feature.

       High Volume + Positive Return: Indicates strong, institutional buying pressure. 
       If a stock's price goes up on significantly higher-than-average volume, 
       the uptrend is widely considered reliable and likely to continue.

       High Volume + Negative Return: Indicates aggressive selling or panic. 
       A sharp price drop accompanied by high volume suggests the downtrend is 
       strong and driven by market conviction.
       
       Low Volume + Positive/Negative Return: Suggests the price move lacks conviction 
       (often driven by retail traders rather than "smart money" institutions). 
       These trends are typically weak and are highly susceptible to reversals or "traps".
       
       return_x_volume = daily_return * Volume"""
    return_df = df.copy()
    return_df.sort_values("Date", inplace=True)

    price_return = return_df.groupby("Ticker")["Close"].diff()
    return_df["return_x_volume"] = return_df["Volume"] * price_return
    return return_df

def create_drawdown_feature(df: pd.DataFrame, window_size: int = 2) -> pd.DataFrame:
    """Create drawdown feature.
       A drawdown is the peak-to-trough decline of an investment.
       Low drawdown magnitude implies stability.
       High drawdown magnitude implies volatility and elevated risk.
       peak = max(Close_t, Close_t-1, ..., Close_t-window_size)
       drawdown = (Close_t - peak) / peak"""
    if window_size <= 1:
        raise ValueError("window_size must be an integer greater than 1.")
    return_df = df.copy()
    return_df.sort_values("Date", inplace=True)
    peak = return_df.groupby("Ticker")["Close"].transform(lambda x: x.rolling(window=window_size, min_periods=window_size).max()).replace(0.0, np.nan)
    return_df[f"drawdown_{window_size}"]= (return_df["Close"] - peak) / peak
    return return_df

def create_rolling_window_mdd_feature(df: pd.DataFrame, window_size: int = 2) -> pd.DataFrame:
    """Create maximum drawdown feature.
       A maximum drawdown is the largest peak-to-trough decline of an investment.
       Low MDD drawdown magnitude implies stability.
       High MDD drawdown magnitude implies volatility and elevated risk.
       peak = max(Close_t, Close_t-1, ..., Close_t-window_size)
       max_drawdown = (drawdown - peak) / peak"""
    if window_size <= 1:
        raise ValueError("window_size must be an integer greater than 1.")
    return_df = df.copy()
    return_df.sort_values("Date", inplace=True)
    peak = return_df.groupby("Ticker")["Close"].transform(lambda x: x.rolling(window=window_size, min_periods=window_size).max()).replace(0.0, np.nan)
    drawdown = ((return_df["Close"] - peak) / peak)
    return_df[f"rolling_window_mdd_{window_size}"] = drawdown.groupby(return_df["Ticker"]).transform(lambda x: x.rolling(window=window_size, min_periods=window_size).min())
    return return_df

def create_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates date feautes: day of week, month."""
    return_df = df.copy()
    return_df.sort_values("Date", inplace=True)
    return_df["day_of_week"] = return_df["Date"].transform(lambda x: x.strftime('%A'))
    return_df["month"] = return_df["Date"].transform(lambda x: x.strftime('%m'))
    return return_df

def create_rolling_sharpe_ratio_feature(df: pd.DataFrame, window_size: int = 2) -> pd.DataFrame:
    """Creates rolling sharpe ratio feature.
       Sharpe ratio measures the performance of an investment compared to a risk-free
       asset after adjusting for its risk.
       Indicates how well the return of an asset compensates for the risk taken.
       When comparing assets, the one with the higher sharpe ratio provides better return 
       relative to the risk required.
       
       rolling_sharpe_ratio = mean_return_window_size / std_return_window_size."""
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