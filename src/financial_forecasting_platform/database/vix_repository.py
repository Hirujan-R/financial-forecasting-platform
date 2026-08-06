from datetime import date
from financial_forecasting_platform.database.connection import get_connection
import pandas as pd

def get_latest_vix_date() -> dict[str, str]:
    connection = get_connection()

    try:
        cursor = connection.cursor()

        cursor.execute(
                f"""
                SELECT MAX(date)
                FROM vix_data;
                """
        )

        result = cursor.fetchone()

        if result[0] is None:
            return None

        return result[0].strftime("%Y-%m-%d")

    finally:
        cursor.close()
        connection.close()



def insert_vix_data(df: pd.DataFrame) -> None:
    """
    Inserts VIX data into vix_data table.

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
                    INSERT INTO vix_data
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
                    ON CONFLICT (date)
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


def get_vix_data_between_dates(
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    Retrieve VIX OHLCV data between two dates.

    Parameters
    ----------
    start_date : str
        Start date in YYYY-MM-DD format.
    end_date : str
        End date in YYYY-MM-DD format.

    Returns
    -------
    pd.DataFrame
        VIX OHLCV data between the dates.
    """

    connection = get_connection()

    query = """
        SELECT *
        FROM vix_data
        WHERE Date BETWEEN %s AND %s
        ORDER BY Date ASC
    """

    try:
        df = pd.read_sql_query(
            query,
            connection,
            params=(start_date, end_date)
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