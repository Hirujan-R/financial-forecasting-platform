import pandas as pd
import pytest

from financial_forecasting_platform.pipelines.data_split.nodes import (
    split_features_target,
    split_train_test,
)


def test_split_features_target():
    df = pd.DataFrame(
        {"feat1": [1, 2, 3], "market_movement": [0, 1, 0]},
        index=pd.date_range("2024-01-01", periods=3),
    )
    result = split_features_target(df, target_variable="market_movement")
    assert "X" in result and "y" in result
    assert "market_movement" not in result["X"].columns
    assert "market_movement" in result["y"].columns


def test_split_train_test():
    X = pd.DataFrame({"feat1": range(10)}, index=pd.date_range("2024-01-01", periods=10))
    y = pd.Series(range(10), index=pd.date_range("2024-01-01", periods=10))

    result = split_train_test(X, y, training_proportion=0.8)
    assert len(result["X_train"]) == 8
    assert len(result["X_test"]) == 2

    with pytest.raises(ValueError):
        split_train_test(X, y, training_proportion=1.5)

