import pandas as pd
import pytest

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