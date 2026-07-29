import numpy as np
import pandas as pd
import pytest
from kedro.pipeline import Pipeline

from financial_forecasting_platform.pipelines.data_validation.nodes import (
    empty_data_validation,
    column_existence_validation,
    missing_data_validation,
    column_type_validation,
    duplicate_validation,
    financial_consistency_validation,
    date_validation,
    ticker_validation,
)
from financial_forecasting_platform.pipelines.data_validation.pipeline import (
    create_pipeline,
)


@pytest.fixture
def valid_ohlcv_df():
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    return pd.DataFrame(
        {
            "Date": dates.tolist() * 2,
            "Ticker": ["AAPL"] * 5 + ["MSFT"] * 5,
            "Open": [150.0, 151.0, 152.0, 153.0, 154.0,
                     300.0, 301.0, 302.0, 303.0, 304.0],
            "High": [152.0, 153.0, 154.0, 155.0, 156.0,
                     302.0, 303.0, 304.0, 305.0, 306.0],
            "Low": [149.0, 150.0, 151.0, 152.0, 153.0,
                    299.0, 300.0, 301.0, 302.0, 303.0],
            "Close": [151.0, 152.0, 153.0, 154.0, 155.0,
                      301.0, 302.0, 303.0, 304.0, 305.0],
            "Volume": [1_000_000, 1_100_000, 1_200_000, 1_300_000, 1_400_000,
                       2_000_000, 2_100_000, 2_200_000, 2_300_000, 2_400_000],
        }
    )


class TestEmptyDataValidation:
    def test_valid_data_passes(self, valid_ohlcv_df):
        result = empty_data_validation(valid_ohlcv_df)
        assert isinstance(result, pd.DataFrame)

    def test_empty_data_raises(self):
        df = pd.DataFrame(columns=["Date", "Ticker", "Open"])
        with pytest.raises(ValueError, match="Data is empty"):
            empty_data_validation(df)

    def test_returns_input_dataframe(self, valid_ohlcv_df):
        result = empty_data_validation(valid_ohlcv_df)
        pd.testing.assert_frame_equal(result, valid_ohlcv_df)


class TestColumnExistenceValidation:
    def test_valid_columns_pass(self, valid_ohlcv_df):
        result = column_existence_validation(valid_ohlcv_df)
        assert isinstance(result, pd.DataFrame)

    def test_missing_columns_raises(self, valid_ohlcv_df):
        df = valid_ohlcv_df.drop(columns=["Volume", "High"])
        with pytest.raises(ValueError, match="Missing required columns"):
            column_existence_validation(df)

    def test_custom_required_columns(self, valid_ohlcv_df):
        result = column_existence_validation(
            valid_ohlcv_df, required_columns=("Date", "Close")
        )
        assert isinstance(result, pd.DataFrame)

    def test_custom_columns_missing_raises(self, valid_ohlcv_df):
        with pytest.raises(ValueError, match="Missing required columns"):
            column_existence_validation(
                valid_ohlcv_df, required_columns=("Date", "Nonexistent")
            )


class TestMissingDataValidation:
    def test_no_missing_data_passes(self, valid_ohlcv_df):
        result = missing_data_validation(valid_ohlcv_df)
        assert isinstance(result, pd.DataFrame)

    def test_missing_data_raises(self, valid_ohlcv_df):
        df = valid_ohlcv_df.copy()
        df.loc[0, "Close"] = np.nan
        with pytest.raises(ValueError, match="Missing values detected"):
            missing_data_validation(df)

    def test_missing_data_in_multiple_columns(self, valid_ohlcv_df):
        df = valid_ohlcv_df.copy()
        df.loc[0, "Close"] = np.nan
        df.loc[1, "Volume"] = np.nan
        with pytest.raises(ValueError, match="Missing values detected"):
            missing_data_validation(df)

    def test_custom_required_columns(self, valid_ohlcv_df):
        result = missing_data_validation(
            valid_ohlcv_df, required_columns=("Date", "Ticker")
        )
        assert isinstance(result, pd.DataFrame)


class TestColumnTypeValidation:
    def test_valid_types_pass(self, valid_ohlcv_df):
        result = column_type_validation(valid_ohlcv_df)
        assert isinstance(result, pd.DataFrame)

    def test_invalid_date_type_raises(self, valid_ohlcv_df):
        df = valid_ohlcv_df.copy()
        df["Date"] = df["Date"].astype(str)
        with pytest.raises(TypeError, match="Invalid 'Date' column type"):
            column_type_validation(df)

    def test_invalid_ticker_type_raises(self, valid_ohlcv_df):
        df = valid_ohlcv_df.copy()
        df["Ticker"] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        with pytest.raises(TypeError, match="Invalid 'Ticker' column type"):
            column_type_validation(df)

    def test_invalid_close_type_raises(self, valid_ohlcv_df):
        df = valid_ohlcv_df.copy()
        df["Close"] = df["Close"].astype(str)
        with pytest.raises(TypeError, match="column type"):
            column_type_validation(df)

    def test_mismatched_columns_and_types_raises(self, valid_ohlcv_df):
        with pytest.raises(ValueError, match="must have equal length"):
            column_type_validation(
                valid_ohlcv_df,
                required_columns=("Date", "Ticker"),
                column_types=("datetime",),
            )

    def test_unsupported_column_type_raises(self, valid_ohlcv_df):
        with pytest.raises(ValueError, match="Unsupported column type"):
            column_type_validation(
                valid_ohlcv_df,
                required_columns=("Date",),
                column_types=("boolean",),
            )

    def test_mismatched_columns_and_error_messages_raises(self, valid_ohlcv_df):
        with pytest.raises(ValueError, match="must have equal length"):
            column_type_validation(
                valid_ohlcv_df,
                required_columns=("Date",),
                column_types=("datetime",),
                error_messages=("msg1", "msg2"),
            )


class TestDuplicateValidation:
    def test_no_duplicates_passes(self, valid_ohlcv_df):
        result = duplicate_validation(valid_ohlcv_df)
        assert isinstance(result, pd.DataFrame)

    def test_duplicates_raises(self, valid_ohlcv_df):
        df = pd.concat([valid_ohlcv_df, valid_ohlcv_df.iloc[:1]], ignore_index=True)
        with pytest.raises(ValueError, match="Duplicate values detected"):
            duplicate_validation(df)

    def test_custom_row_id(self, valid_ohlcv_df):
        result = duplicate_validation(
            valid_ohlcv_df, row_id=("Ticker", "Date")
        )
        assert isinstance(result, pd.DataFrame)

    def test_invalid_row_id_length_raises(self, valid_ohlcv_df):
        with pytest.raises(ValueError, match="exactly two column names"):
            duplicate_validation(valid_ohlcv_df, row_id=("Date",))


class TestFinancialConsistencyValidation:
    def test_valid_data_passes(self, valid_ohlcv_df):
        result = financial_consistency_validation(valid_ohlcv_df)
        assert isinstance(result, pd.DataFrame)

    def test_negative_open_raises(self, valid_ohlcv_df):
        df = valid_ohlcv_df.copy()
        df.loc[0, "Open"] = -1.0
        with pytest.raises(ValueError, match="Open prices must be greater than 0"):
            financial_consistency_validation(df)

    def test_negative_close_raises(self, valid_ohlcv_df):
        df = valid_ohlcv_df.copy()
        df.loc[0, "Close"] = -1.0
        with pytest.raises(ValueError, match="Close prices must be greater than 0"):
            financial_consistency_validation(df)

    def test_high_less_than_open_raises(self, valid_ohlcv_df):
        df = valid_ohlcv_df.copy()
        df.loc[0, "High"] = df.loc[0, "Open"] - 1.0
        with pytest.raises(ValueError, match="High.*Open"):
            financial_consistency_validation(df)

    def test_high_less_than_close_raises(self, valid_ohlcv_df):
        df = valid_ohlcv_df.copy()
        df.loc[0, "High"] = df.loc[0, "Close"] - 1.0
        with pytest.raises(ValueError, match="High.*Close"):
            financial_consistency_validation(df)

    def test_high_less_than_low_raises(self, valid_ohlcv_df):
        df = valid_ohlcv_df.copy()
        df.loc[0, "Low"] = 155.0
        df.loc[0, "High"] = 154.0
        with pytest.raises(ValueError, match="High.*Low"):
            financial_consistency_validation(df)

    def test_low_greater_than_open_raises(self, valid_ohlcv_df):
        df = valid_ohlcv_df.copy()
        df.loc[0, "Low"] = df.loc[0, "Open"] + 1.0
        with pytest.raises(ValueError, match="Low.*Open"):
            financial_consistency_validation(df)

    def test_low_greater_than_close_raises(self, valid_ohlcv_df):
        df = valid_ohlcv_df.copy()
        df.loc[0, "Close"] = 148.0
        df.loc[0, "Low"] = 149.0
        with pytest.raises(ValueError, match="Low.*Close"):
            financial_consistency_validation(df)

    def test_negative_volume_raises(self, valid_ohlcv_df):
        df = valid_ohlcv_df.copy()
        df.loc[0, "Volume"] = -1
        with pytest.raises(ValueError, match="Volume.*greater than or equal to 0"):
            financial_consistency_validation(df)

    def test_invalid_ohlcv_length_raises(self, valid_ohlcv_df):
        with pytest.raises(ValueError, match="exactly five column names"):
            financial_consistency_validation(
                valid_ohlcv_df, ohlcv_columns=("Open", "High", "Low")
            )


class TestDateValidation:
    def test_valid_dates_passes(self, valid_ohlcv_df):
        result = date_validation(valid_ohlcv_df, row_id=["Ticker", "Date"])
        assert isinstance(result, pd.DataFrame)

    def test_future_dates_raises(self, valid_ohlcv_df):
        df = valid_ohlcv_df.copy()
        df.loc[df.index[-1], "Date"] = pd.Timestamp("2099-01-01")
        with pytest.raises(ValueError, match="Future dates detected"):
            date_validation(df, row_id=["Ticker", "Date"])

    def test_start_date_filter(self, valid_ohlcv_df):
        with pytest.raises(ValueError, match="earlier than"):
            date_validation(valid_ohlcv_df, start_date="2024-01-10",
                            row_id=["Ticker", "Date"])

    def test_end_date_filter(self, valid_ohlcv_df):
        with pytest.raises(ValueError, match="later than"):
            date_validation(valid_ohlcv_df, end_date="2024-01-01",
                            row_id=["Ticker", "Date"])

    def test_unsorted_data_raises(self, valid_ohlcv_df):
        df = valid_ohlcv_df.sample(frac=1, random_state=42).reset_index(drop=True)
        with pytest.raises(ValueError, match="not sorted chronologically"):
            date_validation(df, row_id=["Ticker", "Date"])

    def test_invalid_row_id_length_raises(self, valid_ohlcv_df):
        with pytest.raises(ValueError, match="exactly two column names"):
            date_validation(valid_ohlcv_df, row_id=("Date",))


class TestTickerValidation:
    def test_valid_tickers_passes(self, valid_ohlcv_df):
        result = ticker_validation(valid_ohlcv_df)
        assert isinstance(result, pd.DataFrame)

    def test_invalid_ticker_raises(self, valid_ohlcv_df):
        df = valid_ohlcv_df.copy()
        df.loc[0, "Ticker"] = "TSLA"
        with pytest.raises(ValueError, match="Unexpected tickers found"):
            ticker_validation(df)

    def test_custom_tickers(self, valid_ohlcv_df):
        result = ticker_validation(
            valid_ohlcv_df, tickers=("AAPL", "MSFT", "GOOG")
        )
        assert isinstance(result, pd.DataFrame)


class TestDataValidationPipeline:
    def test_returns_pipeline(self):
        result = create_pipeline()
        assert isinstance(result, Pipeline)

    def test_pipeline_has_three_nodes(self):
        pipeline = create_pipeline()
        assert len(pipeline.nodes) == 3

    def test_pipeline_node_names(self):
        pipeline = create_pipeline()
        node_names = {n.name for n in pipeline.nodes}
        expected_names = {
            "stock_data_validation",
            "spy_data_validation",
            "vix_data_validation",
        }
        assert node_names == expected_names

    def test_pipeline_starts_with_raw_data(self):
        pipeline = create_pipeline()
        input_datasets = set()
        for n in pipeline.nodes:
            input_datasets.update(n.inputs)
        assert "raw_data" in input_datasets

    def test_pipeline_outputs_validated_raw_data(self):
        pipeline = create_pipeline()
        output_datasets = set()
        for n in pipeline.nodes:
            output_datasets.update(n.outputs)
        assert "validated_raw_data" in output_datasets
