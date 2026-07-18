from kedro.pipeline import Pipeline, Node
from .nodes import download_ohlcv_data

def create_pipeline(**kwargs) -> Pipeline:
    """Kedro pipeline that takes tickers, start_date, and end_date inputs and calls
    the data_ingestion node."""
    return Pipeline(
        [
            Node(
                func=download_ohlcv_data,
                inputs=[
                    "params:tickers",
                    "params:raw_data_start_date",
                    "params:raw_data_end_date",
                ],
                outputs="raw_data",
                name="data_ingestion_node"

            )
        ]
    )