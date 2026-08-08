import uuid
from datetime import datetime, timezone

from financial_forecasting_platform.database.connection import get_connection


def insert_prediction_log(  # noqa: PLR0913
    ticker: str,
    prediction: int,
    probability: float,
    model_name: str,
    model_version: int | str,
    actual_outcome: int | None = None,
    correct: bool | None = None,
) -> None:
    """
    Inserts a single prediction record into the prediction_logs table.

    Parameters
    ----------
    ticker : str
        Stock symbol the prediction was made for.
    prediction : int
        Predicted class (0 = contraction, 1 = expansion).
    probability : float
        Predicted probability of the expansion class.
    model_name : str
        Name of the registered model that produced the prediction.
    model_version : int | str
        Version of the registered model.
    actual_outcome : int | None
        Known realised outcome, if available.
    correct : bool | None
        Whether the prediction matched the actual outcome, if known.
    """

    connection = get_connection()

    try:
        cursor = connection.cursor()

        cursor.execute(
            """
            INSERT INTO prediction_logs (
                prediction_id,
                timestamp,
                ticker,
                prediction,
                probability,
                model_name,
                model_version,
                actual_outcome,
                correct
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                str(uuid.uuid4()),
                datetime.now(timezone.utc),
                ticker,
                prediction,
                probability,
                model_name,
                model_version,
                actual_outcome,
                correct,
            ),
        )

        connection.commit()

    except Exception:
        connection.rollback()
        raise

    finally:
        cursor.close()
        connection.close()


def get_prediction_logs(limit: int = 50) -> list[dict]:
    """
    Retrieves the most recent prediction logs, newest first.

    Parameters
    ----------
    limit : int
        Maximum number of log rows to return.

    Returns
    -------
    list[dict]
        Prediction log rows keyed by column name.
    """

    connection = get_connection()

    try:
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                prediction_id,
                timestamp,
                ticker,
                prediction,
                probability,
                model_name,
                model_version,
                actual_outcome,
                correct
            FROM prediction_logs
            ORDER BY timestamp DESC
            LIMIT %s
            """,
            (limit,),
        )

        columns = [description[0] for description in cursor.description]

        records = []
        for row in cursor.fetchall():
            record = dict(zip(columns, row))
            record["prediction_id"] = str(record["prediction_id"])
            records.append(record)

        return records

    finally:
        cursor.close()
        connection.close()
