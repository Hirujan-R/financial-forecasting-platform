from kedro.pipeline import Pipeline, Node
from .nodes import download_ohlcv_data, download_market_data, create_spy_data, create_vix_data

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
                name="download_stock_data_node"

            ),
            Node(
                func=download_market_data,
                inputs=[
                    "params:raw_data_start_date",
                    "params:raw_data_end_date",
                    "params:market_tickers"
                ],
                outputs="market_data",
                name="download_market_data_node"
            ),
            Node(
                func=create_spy_data,
                inputs=[
                    "market_data"
                ],
                outputs="raw_spy_data",
                name="create_spy_data_node"
            ),
            Node(
                func=create_vix_data,
                inputs=[
                    "market_data"
                ],
                outputs="raw_vix_data",
                name="create_vix_data_node"
            )
        ]
    )