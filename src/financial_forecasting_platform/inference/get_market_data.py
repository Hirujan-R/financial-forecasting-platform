from datetime import datetime

import pandas as pd

from financial_forecasting_platform.database.spy_repository import (
    get_spy_data_between_dates,
)
from financial_forecasting_platform.database.stock_repository import (
    get_stock_data_between_dates,
)
from financial_forecasting_platform.database.vix_repository import (
    get_vix_data_between_dates,
)
from financial_forecasting_platform.pipelines.data_ingestion.nodes import (
    create_spy_data,
    create_vix_data,
    download_market_data,
    download_ohlcv_data,
    save_spy_data_to_database,
    save_stock_data_to_database,
    save_vix_data_to_database,
)


def _download_stock_fill(ticker: str, start_date, end_date) -> pd.DataFrame:
    """Download OHLCV for a fill window, tolerating windows with no trading days.

    ``download_ohlcv_data`` raises when the requested window is empty (e.g. a
    backfill window that only spans a weekend). Fill windows are best-effort:
    an empty result simply means there is nothing new to insert.
    """
    try:
        return download_ohlcv_data(
            tickers=[ticker],
            start_date=start_date,
            end_date=end_date,
        )
    except ValueError:
        return pd.DataFrame()

def _ensure_stock_data(
    ticker: str,
    start_date: datetime,
    end_date: datetime,
) -> pd.DataFrame:
    """
    Ensures stock OHLCV data exists in the database for the requested
    date range. Any missing data is downloaded from Yahoo Finance,
    inserted into the database, and the complete dataset is returned.
    """

    stock_df = get_stock_data_between_dates(
        tickers=[ticker],
        start_date=start_date,
        end_date=end_date,
    )

    if stock_df.empty:

        downloaded_df = download_ohlcv_data(
            tickers=[ticker],
            start_date=start_date,
            end_date=end_date,
        )

        save_stock_data_to_database(downloaded_df)

    else:

        stock_df["Date"] = pd.to_datetime(stock_df["Date"])

        earliest = stock_df["Date"].min()
        latest = stock_df["Date"].max()

        if earliest > start_date:

            missing_df = _download_stock_fill(
                ticker,
                start_date,
                earliest,
            )

            if not missing_df.empty:
                save_stock_data_to_database(missing_df)

        if latest < end_date:

            missing_df = _download_stock_fill(
                ticker,
                latest,
                end_date,
            )

            if not missing_df.empty:
                save_stock_data_to_database(missing_df)

    return get_stock_data_between_dates(
        tickers=[ticker],
        start_date=start_date,
        end_date=end_date,
    )


def _ensure_spy_data(
    start_date: datetime,
    end_date: datetime,
) -> pd.DataFrame:

    start_date = pd.Timestamp(start_date).normalize()
    end_date = pd.Timestamp(end_date).normalize()

    spy_df = get_spy_data_between_dates(start_date, end_date)

    if spy_df.empty:

        downloaded_df = download_market_data(
            start_date=start_date,
            end_date=end_date,
            tickers=["SPY"]
        )

        if not downloaded_df.empty:
            save_spy_data_to_database(
                create_spy_data(downloaded_df)
            )

    else:

        spy_df["Date"] = pd.to_datetime(spy_df["Date"])

        earliest = spy_df["Date"].min().normalize()
        latest = spy_df["Date"].max().normalize()

        if earliest > start_date:

            missing_df = download_market_data(
                start_date=start_date,
                end_date=earliest,
                tickers=["SPY"]
            )

            if not missing_df.empty:
                save_spy_data_to_database(
                    create_spy_data(missing_df)
                )

        if latest < end_date:

            missing_df = download_market_data(
                start_date=latest + pd.Timedelta(days=1),
                end_date=end_date,
                tickers=["SPY"]
            )

            if not missing_df.empty:
                save_spy_data_to_database(
                    create_spy_data(missing_df)
                )

    return get_spy_data_between_dates(
        start_date,
        end_date
    )


def _ensure_vix_data(
    start_date: datetime,
    end_date: datetime,
) -> pd.DataFrame:


    vix_df = get_vix_data_between_dates(start_date, end_date)

    if vix_df.empty:

        downloaded_df = download_market_data(
            start_date=start_date,
            end_date=end_date,
            tickers=["^VIX"]
        )

        if not downloaded_df.empty:
            save_vix_data_to_database(
                create_vix_data(downloaded_df)
            )

    else:

        vix_df["Date"] = pd.to_datetime(vix_df["Date"])

        earliest = vix_df["Date"].min()
        latest = vix_df["Date"].max()

        if earliest > start_date:
            missing_df = download_market_data(
                start_date=start_date,
                end_date=earliest - pd.Timedelta(days=1),
                tickers=["^VIX"]
            )

            if not missing_df.empty:
                save_vix_data_to_database(create_vix_data(missing_df))

        if latest < end_date:
            missing_df = download_market_data(
                start_date=latest + pd.Timedelta(days=1),
                end_date=end_date,
                tickers=["^VIX"]
            )

            if not missing_df.empty:
                save_vix_data_to_database(create_vix_data(missing_df))

    return get_vix_data_between_dates(start_date, end_date)


def get_market_data(
    ticker: str,
    lookback_window: int = 500,
) -> list[pd.DataFrame]:

    if not isinstance(ticker, str):
        raise TypeError("ticker must be a string.")

    if not isinstance(lookback_window, int):
        raise TypeError("lookback_window must be a positive integer.")

    if lookback_window < 1:
        raise ValueError("lookback_window must be greater than 0.")

    end_date = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
    start_date = end_date - pd.Timedelta(days=lookback_window)

    stock_df = _ensure_stock_data(
        ticker,
        start_date,
        end_date,
    )

    spy_df = _ensure_spy_data(
        start_date,
        end_date,
    )

    vix_df = _ensure_vix_data(
        start_date,
        end_date,
    )

    return [
        stock_df,
        spy_df,
        vix_df,
    ]
