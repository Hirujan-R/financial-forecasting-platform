from datetime import date
from financial_forecasting_platform.database.connection import get_connection
import pandas as pd
from datetime import datetime
from typing import List

def get_latest_stock_dates(tickers: list[str]) -> dict[str, str]:
    connection = get_connection()

    try:
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT Ticker, MAX(Date)
            FROM stock_data
            WHERE Ticker = ANY(%s)
            GROUP BY Ticker;
            """,
            (tickers,)
        )

        results = cursor.fetchall()

        return {
            ticker: latest_date.strftime("%Y-%m-%d")
            for ticker, latest_date in results
        }

    finally:
        cursor.close()
        connection.close()



def insert_stock_data(df: pd.DataFrame) -> None:
    """
    Inserts OHLCV stock data into stock_data table.

    Existing (ticker, date) combinations are ignored.
    """

    if df.empty:
        return

    connection = get_connection()

    try:
        cursor = connection.cursor()

        records = [
            (
                row["Ticker"],
                row["Date"],
                row["Open"],
                row["High"],
                row["Low"],
                row["Close"],
                row["Volume"],
            )
            for _, row in df.iterrows()
        ]

        query = """
                    INSERT INTO stock_data
                    (
                        Ticker,
                        Date,
                        Open,
                        High,
                        Low,
                        Close,
                        Volume
                    )
                    VALUES
                    (
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s
                    )
                    ON CONFLICT (ticker, date)
                    DO NOTHING;
                """

        cursor.executemany(
            query,
            records
        )

        connection.commit()

    except Exception:
        connection.rollback()
        raise

    finally:
        cursor.close()
        connection.close()


def get_stock_data_between_dates(
    tickers: List[str],
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    Retrieve stock OHLCV data for specified tickers between two dates.

    Parameters
    ----------
    tickers : List[str]
        List of ticker symbols.
    start_date : str
        Start date in YYYY-MM-DD format.
    end_date : str
        End date in YYYY-MM-DD format.

    Returns
    -------
    pd.DataFrame
        Stock OHLCV data.
    """

    connection = get_connection()

    placeholders = ",".join(["%s"] * len(tickers))

    query = f"""
        SELECT *
        FROM stock_data
        WHERE Ticker IN ({placeholders})
        AND Date BETWEEN %s AND %s
        ORDER BY Ticker, Date ASC
    """

    params = [
        *tickers,
        start_date,
        end_date
    ]

    try:
        df = pd.read_sql_query(
            query,
            connection,
            params=params
        )
        df.columns = df.columns.str.capitalize()
        df["Ticker"] = df["Ticker"].astype("string")
        df["Date"] = pd.to_datetime(df["Date"])
        df["Open"] = df["Open"].astype("float64")
        df["High"] = df["High"].astype("float64")
        df["Low"] = df["Low"].astype("float64")
        df["Close"] = df["Close"].astype("float64")
        df["Volume"] = df["Volume"].astype("int64")
        return df

    finally:
        connection.close()