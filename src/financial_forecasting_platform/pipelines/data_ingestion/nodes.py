import pandas as pd
import yfinance as yf
from datetime import timedelta
from financial_forecasting_platform.database.stock_repository import \
    get_latest_stock_dates, insert_stock_data, get_stock_data_between_dates
from financial_forecasting_platform.database.spy_repository import \
    get_latest_spy_date, insert_spy_data, get_spy_data_between_dates
from financial_forecasting_platform.database.vix_repository import \
    get_latest_vix_date, insert_vix_data, get_vix_data_between_dates


from datetime import datetime, timedelta


def get_earliest_stock_download_start_date(
    tickers: list[str],
    default_start_date: str,
    end_date: str
) -> str:
    latest_dates = get_latest_stock_dates(tickers)

    downloads = {}

    for ticker in tickers:

        if ticker not in latest_dates:
            start_date = default_start_date

        else:
            latest_date = datetime.strptime(
                latest_dates[ticker],
                "%Y-%m-%d"
            )

            start_date = (
                latest_date + timedelta(days=1)
            ).strftime("%Y-%m-%d")

        if start_date < end_date:
            downloads[ticker] = start_date
    if not downloads:
        return ""
    return min(downloads.values())

def get_earliest_market_download_start_date(
    default_start_date: str,
    end_date: str
) -> str:

    latest_spy_date = get_latest_spy_date()
    latest_vix_date = get_latest_vix_date()

    if latest_spy_date is None:
        spy_start_date = default_start_date
    else:
        spy_start_date = (
            datetime.strptime(latest_spy_date, "%Y-%m-%d")
            + timedelta(days=1)
        ).strftime("%Y-%m-%d")

    if latest_vix_date is None:
        vix_start_date = default_start_date
    else:
        vix_start_date = (
            datetime.strptime(latest_vix_date, "%Y-%m-%d")
            + timedelta(days=1)
        ).strftime("%Y-%m-%d")

    earliest_start_date = min(spy_start_date, vix_start_date)

    if earliest_start_date >= end_date:
        return ""

    return earliest_start_date

def download_ohlcv_data(tickers: list[str], start_date: str, end_date: str
                        ) -> pd.DataFrame:
    """Downloads OHLCV for a list "tickers" from "start_date" to "end_date" using
       yfinance api. "Date" and "Ticker" indices are reset to columns."""
    if start_date == "":
        return pd.DataFrame()
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

def save_stock_data_to_database(df):
    """
    Saves downloaded OHLCV data into PostgreSQL.
    """

    insert_stock_data(df)

def download_market_data(
    start_date: str,
    end_date: str,
    tickers: list[str] | None = None
) -> pd.DataFrame:
    """Downloads data from yfinance from general markets."""

    if start_date == "":
        return pd.DataFrame()

    if tickers is None:
        tickers = ["SPY", "^VIX"]

    df = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=False,
    )

    if df.empty:
        return pd.DataFrame()

    # Handle multi-ticker download
    if isinstance(df.columns, pd.MultiIndex):

        df = (
            df
            .stack(level="Ticker", future_stack=True)
            .reset_index()
        )

    # Handle single ticker download
    else:

        df = df.reset_index()
        df["Ticker"] = tickers[0]

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
    if df.empty:
        return pd.DataFrame()
    return_df = df.copy()
    return_df["Volume"] = return_df["Volume"].fillna(0).astype("int64")
    return return_df[return_df["Ticker"] == "SPY"]

def save_spy_data_to_database(df):
    """
    Saves downloaded SPY data into PostgreSQL.
    """

    insert_spy_data(df)

def create_vix_data(df: pd.DataFrame) -> pd.DataFrame:
    """Retrieves VIX data from market_data"""
    if df.empty:
        return pd.DataFrame()
    return_df = df.copy()
    return_df["Volume"] = return_df["Volume"].fillna(0).astype("int64")
    return return_df[return_df["Ticker"] == "^VIX"]

def save_vix_data_to_database(df):
    """
    Saves downloaded VIX data into PostgreSQL.
    """

    insert_vix_data(df)

def get_stock_data(tickers, start_date, end_date):
    return get_stock_data_between_dates(tickers, start_date, end_date)

def get_spy_data(start_date, end_date):
    return get_spy_data_between_dates(start_date, end_date)

def get_vix_data(start_date, end_date):
    return get_vix_data_between_dates(start_date, end_date)