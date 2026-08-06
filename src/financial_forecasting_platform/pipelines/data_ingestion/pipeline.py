from kedro.pipeline import Pipeline, Node
from .nodes import get_earliest_stock_download_start_date, download_ohlcv_data, \
    save_stock_data_to_database, get_earliest_market_download_start_date, \
    download_market_data, create_spy_data, save_spy_data_to_database, \
    create_vix_data, save_vix_data_to_database, get_stock_data, get_spy_data, \
    get_vix_data

def create_pipeline(**kwargs) -> Pipeline:
    """Kedro pipeline that takes tickers, start_date, and end_date inputs and calls
    the data_ingestion node."""
    return Pipeline(
        [   
            Node(
                func=get_earliest_stock_download_start_date,
                inputs=[
                    "params:tickers",
                    "params:raw_data_start_date",
                    "params:raw_data_end_date",
                ],
                outputs="stock_download_start_date",
                name="get_stock_download_start_dates_node"

            ),
            Node(
                func=download_ohlcv_data,
                inputs=[
                    "params:tickers",
                    "stock_download_start_date",
                    "params:raw_data_end_date",
                ],
                outputs="stock_db_data",
                name="download_stock_data_node"

            ),
            Node(
                func=save_stock_data_to_database,
                inputs=[
                    "stock_db_data",
                ],
                outputs=None,
                name="insert_stock_data_to_database_node"

            ),
            Node(
                func=get_earliest_market_download_start_date,
                inputs=[
                    "params:raw_data_start_date",
                    "params:raw_data_end_date",
                ],
                outputs="market_download_start_date",
                name="get_market_download_start_date_node"

            ),
            Node(
                func=download_market_data,
                inputs=[
                    "market_download_start_date",
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
                outputs="spy_db_data",
                name="create_spy_data_node"
            ),
            Node(
                func=save_spy_data_to_database,
                inputs=[
                    "spy_db_data"
                ],
                outputs=None,
                name="insert_spy_data_to_database_node"
            ),
            Node(
                func=create_vix_data,
                inputs=[
                    "market_data"
                ],
                outputs="vix_db_data",
                name="create_vix_data_node"
            ),
            Node(
                func=save_vix_data_to_database,
                inputs=[
                    "vix_db_data"
                ],
                outputs=None,
                name="insert_vix_data_to_database_node"
            ),
            Node(
                func=get_stock_data,
                inputs=[
                    "params:tickers",
                    "params:raw_data_start_date",
                    "params:raw_data_end_date"
                ],
                outputs="raw_data",
                name="get_stock_data_from_database_node"
            ),
            Node(
                func=get_spy_data,
                inputs=[
                    "params:raw_data_start_date",
                    "params:raw_data_end_date"
                ],
                outputs="raw_spy_data",
                name="get_spy_data_from_database_node"
            ),
            Node(
                func=get_vix_data,
                inputs=[
                    "params:raw_data_start_date",
                    "params:raw_data_end_date"
                ],
                outputs="raw_vix_data",
                name="get_vix_data_from_database_node"
            )
        ]
    )