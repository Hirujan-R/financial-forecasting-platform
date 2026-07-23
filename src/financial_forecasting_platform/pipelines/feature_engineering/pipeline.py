from kedro.pipeline import Pipeline, Node

from .nodes import create_all_features, lr_feature_engineering, \
    xgboost_feature_engineering, mlp_feature_engineering


def create_pipeline(**kwargs):
    """Kedro pipeline that takes cleaned dataset as input and calls
    the feature_engineering node to generate features."""
    return Pipeline(
        [
            Node(
                func=create_all_features,
                inputs=["validated_raw_data", "params:create_all_features_config", 
                        "params:columns_to_drop", "params:date_column"],
                outputs="all_features",
                name="create_all_features_node"
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