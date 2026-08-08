import os

import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _isolate_mlflow_env():
    """Restores MLflow-related env vars after each test.

    Kedro's kedro-mlflow plugin sets ``MLFLOW_TRACKING_URI`` when a session
    runs; without isolation that leaks into later tests and makes them hit an
    external/irrelevant MLflow server.
    """
    keys = [key for key in os.environ if key.startswith("MLFLOW_")]
    saved = {key: os.environ[key] for key in keys}
    try:
        yield
    finally:
        for key in keys:
            os.environ.pop(key, None)
        os.environ.update(saved)

@pytest.fixture
def sample_ohlcv():
    dates = pd.date_range("2024-01-01", periods=5)

    return pd.DataFrame(
        {
            "Date": dates.tolist() * 2,
            "Ticker": ["A"] * 5 + ["B"] * 5,
            "Open": [10, 11, 12, 13, 14,
                     20, 21, 22, 23, 24],
            "High": [11, 12, 13, 14, 15,
                     21, 22, 23, 24, 25],
            "Low": [9, 10, 11, 12, 13,
                    19, 20, 21, 22, 23],
            "Close": [10, 11, 12, 13, 14,
                      20, 21, 22, 23, 24],
            "Volume": [100, 110, 120, 130, 140,
                       200, 210, 220, 230, 240],
        }
    ).set_index("Date")
