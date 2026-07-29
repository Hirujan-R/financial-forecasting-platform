from kedro.pipeline import Pipeline, Node

from .nodes import create_all_features, lr_feature_engineering, \
    xgboost_feature_engineering, mlp_feature_engineering, merge_dataframes


def create_pipeline(**kwargs):
    """Kedro pipeline that takes cleaned dataset as input and calls
    the feature_engineering node to generate features."""
    return Pipeline(
        [
            Node(
                func=create_all_features,
                inputs=["validated_raw_data", "params:stock_features_config", 
                        "params:columns_to_drop", "params:date_column"],
                outputs="stock_features",
                name="create_stock_features_node"
            ),
            Node(
                func=create_all_features,
                inputs=["validated_spy_data", "params:spy_features_config", 
                        "params:spy_columns_to_drop", "params:date_column"],
                outputs="spy_features",
                name="create_spy_features_node"
            ),
            Node(
                func=create_all_features,
                inputs=["validated_vix_data", "params:vix_features_config", 
                        "params:vix_columns_to_drop", "params:date_column"],
                outputs="vix_features",
                name="create_vix_features_node"
            ),
            Node(
                func=merge_dataframes,
                inputs=["stock_features", "spy_features", 
                        "vix_features"],
                outputs="all_features",
                name="merge_data_node"
            ),
            Node(
                func=lr_feature_engineering,
                inputs=["all_features", "params:lr_features"],
                outputs="lr_features",
                name="lr_feature_engineering_node",
            ),
            Node(
                func=xgboost_feature_engineering,
                inputs=["all_features", "params:xgboost_features"],
                outputs="xgboost_features",
                name="xgboost_feature_engineering_node",
            ),
            Node(
                func=mlp_feature_engineering,
                inputs=["all_features", "params:mlp_features"],
                outputs="mlp_features",
                name="mlp_feature_engineering_node",
            )
        ]
    )