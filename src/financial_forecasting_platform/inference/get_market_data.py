import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


def get_market_data(ticker: str, lookback_window: int = 500) -> pd.DataFrame:

    if not isinstance(ticker, str):
        raise TypeError("ticker must be a string.")
    if not isinstance(lookback_window, int):
        raise TypeError("lookback_window must be a positive integer.")
    if lookback_window < 1:
        raise ValueError("lookback_window must be an integer greater than 1.")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_window)
    tickers = [ticker, "SPY", "^VIX"]
    dfs = []
    for ticker in tickers:
        df = yf.download(
            ticker,
            group_by="Ticker",
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            repair=True
        )

        if df.empty:
            raise ValueError(
                "No data returned from Yahoo Finance"
            )

        df = df.stack(
            level=0,
            future_stack=True
        )

        df = df.reset_index()

        df = df.sort_values(["Ticker", "Date"])
        dfs.append(df)

    return dfs
