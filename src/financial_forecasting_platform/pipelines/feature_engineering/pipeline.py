from kedro.pipeline import Pipeline, node

from .nodes import lr_feature_engineering, xgboost_feature_engineering, mlp_feature_engineering


def create_pipeline(**kwargs):
    """Kedro pipeline that takes cleaned dataset as input and calls
    the feature_engineering node to generate features."""
    return Pipeline(
        [
            node(
                func=lr_feature_engineering,
                inputs="validated_raw_data",
                outputs="lr_features",
                name="lr_feature_engineering_node",
            ),
            node(
                func=xgboost_feature_engineering,
                inputs="validated_raw_data",
                outputs="xgboost_features",
                name="xgboost_feature_engineering_node",
            ),
            node(
                func=mlp_feature_engineering,
                inputs="validated_raw_data",
                outputs="mlp_features",
                name="mlp_feature_engineering_node",
            )
        ]
    )