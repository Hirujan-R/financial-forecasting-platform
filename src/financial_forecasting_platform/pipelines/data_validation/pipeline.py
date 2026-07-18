from kedro.pipeline import Node, Pipeline
from.nodes import empty_data_validation, column_existence_validation, \
    missing_data_validation, column_type_validation, duplicate_validation, \
    financial_consistency_validation, date_validation, ticker_validation


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=empty_data_validation,
                inputs=[
                    "raw_data"
                ],
                outputs="validated_non_empty_data",
                name="empty_data_validation_node"
            ),
            Node(
                func=column_existence_validation,
                inputs=[
                    "validated_non_empty_data",
                    "params:required_columns"
                ],
                outputs="validated_schema_data",
                name="column_existence_validation_node"
            ),
            Node(
                func=missing_data_validation,
                inputs=[
                    "validated_schema_data",
                    "params:required_columns"
                ],
                outputs="validated_missing_data",
                name="missing_data_validation_node"
            ),
            Node(
                func=column_type_validation,
                inputs=[
                    "validated_missing_data",
                    "params:required_columns",
                    "params:column_types",
                    "params:error_messages"
                ],
                outputs="validated_type_data",
                name="column_type_validation_node"
            ),
            Node(
                func=duplicate_validation,
                inputs=[
                    "validated_type_data",
                    "params:row_id"
                ],
                outputs="validated_duplicate_data",
                name="duplicate_validation_node"
            ),
            Node(
                func=financial_consistency_validation,
                inputs=[
                    "validated_duplicate_data",
                    "params:ohlcv_columns"
                ],
                outputs="validated_financial_data",
                name="financial_consistency_validation_node"
            ),
            Node(
                func=date_validation,
                inputs=[
                    "validated_financial_data",
                    "params:raw_data_start_date",
                    "params:raw_data_end_date",
                    "params:row_id",
                    "params:max_gap_days"
                ],
                outputs="validated_date_data",
                name="date_validation_node"
            ),
            Node(
                func=ticker_validation,
                inputs=[
                    "validated_date_data",
                    "params:tickers"
                ],
                outputs="validated_raw_data",
                name="ticker_validation_node"
            )
        ]
    )
