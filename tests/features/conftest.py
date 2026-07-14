import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_ohlcv_df():
    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    tickers = ["AAPL", "MSFT"]
    np.random.seed(42)
    rows = []
    for ticker in tickers:
        ticker_close = 150 + np.cumsum(np.random.randn(len(dates)) * 2)
        ticker_df = pd.DataFrame(
            {
                "Date": dates,
                "Ticker": ticker,
                "Open": ticker_close + np.random.randn(len(dates)) * 0.5,
                "High": ticker_close + abs(np.random.randn(len(dates))),
                "Low": ticker_close - abs(np.random.randn(len(dates))),
                "Close": ticker_close,
                "Volume": np.random.randint(1_000_000, 10_000_000, size=len(dates)),
            }
        )
        rows.append(ticker_df)
    df = pd.concat(rows, ignore_index=True)
    df["High"] = df[["Open", "Close", "High"]].max(axis=1)
    df["Low"] = df[["Open", "Close", "Low"]].min(axis=1)
    df.set_index("Date", inplace=True)
    return df


@pytest.fixture
def single_ticker_df():
    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    np.random.seed(99)
    n = len(dates)
    close = 100 + np.cumsum(np.random.randn(n) * 1.5)
    df = pd.DataFrame(
        {
            "Ticker": "AAPL",
            "Open": close + np.random.randn(n) * 0.3,
            "High": close + abs(np.random.randn(n)),
            "Low": close - abs(np.random.randn(n)),
            "Close": close,
            "Volume": np.random.randint(500_000, 5_000_000, size=n),
        },
        index=dates,
    )
    df.index.name = "Date"
    df["High"] = df[["Open", "Close", "High"]].max(axis=1)
    df["Low"] = df[["Open", "Close", "Low"]].min(axis=1)
    return df
