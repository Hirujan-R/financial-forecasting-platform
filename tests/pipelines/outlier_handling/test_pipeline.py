import pandas as pd

from financial_forecasting_platform.pipelines.outlier_handling.nodes import (
    clip_outliers,
    outlier_detection,
)


def test_outlier_detection():
    df = pd.DataFrame({"feat1": [1.0, 2.0, 100.0, 1.5, 2.1]})
    res = outlier_detection(df, outlier_feature_selection=["feat1"])
    assert "is_outlier" in res.columns
    assert set(res["is_outlier"].unique()).issubset({0, 1})


def test_clip_outliers():
    df = pd.DataFrame({"feat1": range(100)})
    res = clip_outliers(df, clip_columns=["feat1"])
    assert res["feat1"].min() >= df["feat1"].quantile(0.01)
    assert res["feat1"].max() <= df["feat1"].quantile(0.99)

