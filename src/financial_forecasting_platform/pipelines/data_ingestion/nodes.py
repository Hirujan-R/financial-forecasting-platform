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
        end=end_date,
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

    return df

def download_market_data(start_date: str, end_date: str, tickers: list[str] 
                         | None = None) -> pd.DataFrame:
    """Downloads data from yfinance from general markets."""
    if tickers is None:
        tickers = ["SPY", "^VIX"]
    df = yf.download(
    tickers,
    start=start_date,
    end=end_date
    )
    df = (
        df
        .stack(level="Ticker", future_stack=True)
        .reset_index()
    )

    # Reorder columns
    df = df[
        [
            "Date",
            "Ticker",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
        ]
    ]
    return df

def create_spy_data(df: pd.DataFrame) -> pd.DataFrame:
    """Retrieves SPY data from market_data"""
    return df[df["Ticker"] == "SPY"]

def create_vix_data(df: pd.DataFrame) -> pd.DataFrame:
    """Retrieves VIX data from market_data"""
    return df[df["Ticker"] == "^VIX"]