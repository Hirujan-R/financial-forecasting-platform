# ruff: noqa: PLR2004
import uuid
from unittest.mock import patch

import pytest

from financial_forecasting_platform.database.prediction_repository import (
    get_prediction_logs,
    insert_prediction_log,
)


class FakeCursor:
    def __init__(self, fetch_result=None, description=()):
        self.execute_calls = []
        self.fetch_result = fetch_result
        self.description = description

    def execute(self, query, params=None):
        self.execute_calls.append((query, params))

    def fetchall(self):
        return self.fetch_result

    def close(self):
        pass


class FakeConnection:
    def __init__(self, cursor):
        self._cursor = cursor
        self.committed = False
        self.rolled_back = False

    def cursor(self):
        return self._cursor

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True

    def close(self):
        pass


@patch("financial_forecasting_platform.database.prediction_repository.get_connection")
def test_insert_prediction_log(mock_get_connection):
    cursor = FakeCursor()
    connection = FakeConnection(cursor)
    mock_get_connection.return_value = connection

    insert_prediction_log(
        ticker="AAPL",
        prediction=1,
        probability=0.82,
        model_name="XGBoost",
        model_version="3",
        actual_outcome=1,
        correct=True,
    )

    assert len(cursor.execute_calls) == 1
    query, params = cursor.execute_calls[0]
    assert "INSERT INTO prediction_logs" in query

    (prediction_id, timestamp, ticker, prediction, probability,
     model_name, model_version, actual_outcome, correct) = params
    assert ticker == "AAPL"
    assert prediction == 1
    assert probability == 0.82
    assert model_name == "XGBoost"
    assert model_version == "3"
    assert actual_outcome == 1
    assert correct is True
    assert prediction_id
    assert timestamp is not None

    assert connection.committed is True


@patch("financial_forecasting_platform.database.prediction_repository.get_connection")
def test_insert_prediction_log_rolls_back_on_error(mock_get_connection):
    class ExplodingCursor(FakeCursor):
        def execute(self, query, params=None):
            raise RuntimeError("boom")

    connection = FakeConnection(ExplodingCursor())
    mock_get_connection.return_value = connection

    with pytest.raises(RuntimeError, match="boom"):
        insert_prediction_log(
            ticker="AAPL",
            prediction=1,
            probability=0.82,
            model_name="XGBoost",
            model_version="3",
        )

    assert connection.rolled_back is True


@patch("financial_forecasting_platform.database.prediction_repository.get_connection")
def test_get_prediction_logs(mock_get_connection):
    description = [
        ("prediction_id",), ("timestamp",), ("ticker",), ("prediction",),
        ("probability",), ("model_name",), ("model_version",),
        ("actual_outcome",), ("correct",),
    ]
    rows = [
        (
            "uuid-1",
            "2026-08-07 10:00:00",
            "AAPL",
            1,
            0.82,
            "XGBoost",
            3,
            1,
            True,
        )
    ]
    cursor = FakeCursor(fetch_result=rows, description=description)
    connection = FakeConnection(cursor)
    mock_get_connection.return_value = connection

    result = get_prediction_logs(limit=10)

    assert result == [
        {
            "prediction_id": "uuid-1",
            "timestamp": "2026-08-07 10:00:00",
            "ticker": "AAPL",
            "prediction": 1,
            "probability": 0.82,
            "model_name": "XGBoost",
            "model_version": 3,
            "actual_outcome": 1,
            "correct": True,
        }
    ]
    query, params = cursor.execute_calls[0]
    assert "ORDER BY timestamp DESC" in query
    assert params == (10,)


@patch("financial_forecasting_platform.database.prediction_repository.get_connection")
def test_get_prediction_logs_converts_uuid_to_string(mock_get_connection):
    prediction_id = uuid.uuid4()
    description = [
        ("prediction_id",), ("timestamp",), ("ticker",), ("prediction",),
        ("probability",), ("model_name",), ("model_version",),
        ("actual_outcome",), ("correct",),
    ]
    rows = [
        (
            prediction_id,
            "2026-08-07 10:00:00",
            "AAPL",
            1,
            0.82,
            "XGBoost",
            3,
            None,
            None,
        )
    ]
    cursor = FakeCursor(fetch_result=rows, description=description)
    mock_get_connection.return_value = FakeConnection(cursor)

    result = get_prediction_logs()

    assert result[0]["prediction_id"] == str(prediction_id)
    assert isinstance(result[0]["prediction_id"], str)


@patch("financial_forecasting_platform.database.prediction_repository.get_connection")
def test_get_prediction_logs_empty(mock_get_connection):
    description = [("prediction_id",), ("timestamp",)]
    cursor = FakeCursor(fetch_result=[], description=description)
    mock_get_connection.return_value = FakeConnection(cursor)

    assert get_prediction_logs() == []
