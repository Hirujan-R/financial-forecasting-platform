import pandas as pd
import yfinance as yf

def download_ohlcv_data(tickers: list[str], start_date: str, end_date: str
                        ) -> pd.DataFrame:
    """Downloads OHLCV for a list "tickers" from "start_date" to "end_date" using
       yfinance api. "Date" and "Ticker" indices are reset to columns."""
    if not tickers:
        raise ValueError("tickers cannot be empty")

    if not all(isinstance(t, str) for t in tickers):
        raise TypeError("tickers must contain strings")

    if start_date >= end_date:
        raise ValueError(
            "start_date must be before end_date"
        )

    tickers = [ticker.upper() for ticker in tickers]

    df = yf.download(
        tickers,
        group_by="Ticker",
        start=start_date,
        end=end_date
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

    return df