from kedro.pipeline import Pipeline, node
from .nodes import download_ohlcv_data

def create_pipeline(**kwargs) -> Pipeline:

    return Pipeline(
        [
            node(
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