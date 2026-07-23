import pandas as pd
from pandas.api.types import is_datetime64_any_dtype as is_datetime, \
    is_integer_dtype as is_integer, \
    is_numeric_dtype as is_numeric, is_string_dtype as is_string

# Default values for parameters
DEFAULT_REQUIRED_COLUMNS = (
    "Date", "Ticker", "Open", "Close", "High", "Low", "Volume"
)

DEFAULT_COLUMN_TYPES = ("datetime", "string", "numeric", "numeric",
                        "numeric", "numeric", "integer")
TYPE_VALIDATORS = {
    "datetime": is_datetime,
    "integer": is_integer,
    "numeric": is_numeric,
    "string": is_string,
}

DEFAULT_ERROR_MESSAGES = ("Invalid 'Date' column type. 'Date' column should be in DateTime format.",
    "Invalid 'Ticker' column type. 'Ticker' column must contain string values.",
    "Invalid 'Open' column type. 'Open' column must contain numeric floating-point values.",
    "Invalid 'High' column type. 'High' column must contain numeric floating-point values.",
    "Invalid 'Low' column type. 'Low' column must contain numeric floating-point values.",
    "Invalid 'Close' column type. 'Close' column must contain numeric floating-point values.",
    "Invalid 'Volume' column type. 'Volume' column must contain numeric integer values."
)
DEFAULT_ROW_ID = ("Ticker", "Date")
DEFAULT_OHLCV_COLUMNS = ("Open", "High", "Low", "Close", "Volume")
DEFAULT_TICKERS = ("GOOG","AAPL","MSFT","NVDA")


def empty_data_validation(df: pd.DataFrame) -> pd.DataFrame:
    """Raises an error if df is empty."""
    if df.empty:
        raise ValueError("Data is empty. Data must not be empty.")
    return df

def column_existence_validation(df: pd.DataFrame, required_columns: tuple | None = None) \
                                -> pd.DataFrame:
    """Raises an error if df doesn't contain all the required columns."""
    if required_columns is None:
        # Use default values if parameter not specified.
        required_columns = DEFAULT_REQUIRED_COLUMNS
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )
    return df

def missing_data_validation(df: pd.DataFrame, required_columns: tuple | None = None) \
                            -> pd.DataFrame:  
    """Raises an error if there are null values in the required columns of the data."""
    if required_columns is None:
        # Use default values if param not specified.
        required_columns = DEFAULT_REQUIRED_COLUMNS
    error_msgs = []
    for column in required_columns:
        missing_count = df[column].isnull().sum()
        if missing_count > 0:
            error_msgs.append(f"- {column}: {missing_count}")
    if error_msgs:
        error_msg = "Missing values detected:\n" + "\n".join(error_msgs)
        raise ValueError(error_msg)
    return df

    
def column_type_validation(df: pd.DataFrame, required_columns: tuple | None = None, 
                           column_types: tuple | None = None,
                           error_messages: tuple | None = None)  -> pd.DataFrame:
    """Raises an error if the data types of the required columns are not correctly specified."""
    if required_columns is None:
        # Use default values if param not specified.
        required_columns = DEFAULT_REQUIRED_COLUMNS
    if column_types is None:
        # Use default values if param not specified.
        column_types = DEFAULT_COLUMN_TYPES
    # Raises error if there is a length mismatch between required_columns and column_types.
    if len(required_columns) != len(column_types):
        raise ValueError(
            "required_columns and column_types must have equal length."
        )
    for column_type in column_types:
        # Raises error if column type is not valid.
        if column_type not in TYPE_VALIDATORS:
            raise ValueError(
                f"Unsupported column type '{column_type}'. "
                f"Expected one of: {list(TYPE_VALIDATORS.keys())}"
            )
    type_checking_functions = [TYPE_VALIDATORS[column_type] for column_type in column_types]

    expected_type = dict(zip(required_columns,type_checking_functions))

    if error_messages is None:
        # Use default values if param not specified.
        error_messages = DEFAULT_ERROR_MESSAGES
    if len(required_columns) != len(error_messages):
        # Raises error if there is a length mismatch between required_columns and error_messages.
        raise ValueError(
            "required_columns and error_messages must have equal length."
        )
    error_message_dict = dict(zip(required_columns, error_messages))

    for column, validator in expected_type.items():
        # Raises error if type of column is not the same as specified type.
        if not validator(df[column]):
            raise TypeError(error_message_dict[column])
    return df
    

def duplicate_validation(df: pd.DataFrame, row_id: tuple | None = None) -> pd.DataFrame:
    """Raises error if the data contains duplicate values."""
    # row_id contains identifiers of each row.
    if row_id is None:
        # Use default values if param not specified.
        row_id = DEFAULT_ROW_ID
    if len(row_id) != 2:
        # Raise error if number of keys aren't same as specified amount.
        raise ValueError(
            "row_id must contain exactly two column names."
        )
    if df.duplicated(subset=row_id).any():
        # Raises error if any two rows contain same keys.
        raise ValueError("Duplicate values detected in data. Duplicate values are not accepted.")
    return df


def financial_consistency_validation(df: pd.DataFrame, ohlcv_columns: tuple | None = None) \
                                     -> pd.DataFrame:
    """Raises error if financial market data is not consistent."""
    # Adding tolerance to account for yfinance api autoadjust inconsistencies.
    tolerance = 1e-4
    if ohlcv_columns is None:
        # Use default values if param not specified.
        ohlcv_columns = DEFAULT_OHLCV_COLUMNS
    if len(ohlcv_columns) != 5:
        # Raises error if number of specified columns isn't 5 for O-H-L-C-V.
        raise ValueError(
            "ohlcv_columns must contain exactly five column names."
        )
    open_col, high_col, low_col, close_col, volume_col = ohlcv_columns
    if (df[open_col] <= 0).any():
        raise ValueError("Open prices must be greater than 0.")
    if (df[close_col] <= 0).any():
        raise ValueError("Close prices must be greater than 0.")
    if (df[high_col] < (df[open_col] * (1 - tolerance))).any():
        raise ValueError("The High price must be greater than or equal to the Open price.")
    if (df[high_col] < (df[close_col] * (1 - tolerance))).any():
        raise ValueError("The High price of an asset must be greater than or equal to the Close price.")
    if (df[high_col] < (df[low_col] * (1 - tolerance))).any():
        raise ValueError("The High price of an asset must be greater than or equal to the Low price.")
    if (df[low_col] > (df[open_col] * (1 + tolerance))).any():
        raise ValueError("The Low price of an asset must be smaller than or equal to the Open price.")
    if (df[low_col] > (df[close_col] * (1 + tolerance))).any():
        raise ValueError("The Low price of an asset must be smaller than or equal to the Close price.")
    if ((df[volume_col] < 0)).any():
        raise ValueError("The Volume of an asset must be greater than or equal to 0.")
    return df


    # Date Validation
def date_validation(df: pd.DataFrame, start_date: str = None, end_date: str = None, 
                    row_id: tuple | None = None, max_gap_days: int = 10) -> pd.DataFrame:
    """Raises error if invalid date is selected."""
    if row_id is None:
        # Use default values if param not specified.
        row_id = DEFAULT_ROW_ID
    if len(row_id) != 2:
        raise ValueError(
            "row_id must contain exactly two column names."
        )
    ticker, date = row_id
    df_sorted = df.sort_values(row_id)
    if not df.equals(df_sorted):
        raise ValueError(
            f"Data is not sorted chronologically by {row_id[0]} and {row_id[1]}."
        )
    df_sorted[date] = pd.to_datetime(df_sorted[date])
    if df_sorted[date].dt.tz is None:
        df_sorted[date] = df_sorted[date].dt.tz_localize("UTC")
    else:
        df_sorted[date] = df_sorted[date].dt.tz_convert("UTC")
    
    
    today = pd.Timestamp.today(tz='UTC').normalize()
    if (df_sorted[date] > today).any():
        future_dates = df.loc[df_sorted[date] > today, date].unique()

        raise ValueError(
            f"Future dates detected: {future_dates[:5]}"
        )
    
    # Raises error if data contains row with start date outside specified data ranges.
    if start_date:
        start_date = pd.Timestamp(start_date, tz='UTC') 
        if (df_sorted[date] < start_date).any():
            raise ValueError(
                f"Dates earlier than {start_date.date()} detected."
            )
    if end_date:
        end_date = pd.Timestamp(end_date, tz="UTC")
        if (df_sorted[date] > end_date).any():
            raise ValueError(
                f"Dates later than {end_date.date()} detected."
            )
    
    # Raises error if there is an unusually large gap between dates of 2 cconsecutive rows.
    date_diff = (
        df_sorted
        .groupby(ticker)[date]
        .diff()
    )
    if (date_diff > pd.Timedelta(days=max_gap_days)).any():
        raise ValueError(
            "Large gap detected in time series."
        )
    return df
    

def ticker_validation(df: pd.DataFrame, tickers: tuple | None = None) \
                      -> pd.DataFrame:
    """Raises error if the data contains an invalid ticker."""
    if tickers is None:
        # Use default values if param not specified.
        tickers = DEFAULT_TICKERS
    invalid = set(df["Ticker"]) - set(tickers)
    if invalid:
        raise ValueError(
            f"Unexpected tickers found: {sorted(invalid)}. \
                \nTickers must be one of the following values: {tickers}"
        )
    return df
