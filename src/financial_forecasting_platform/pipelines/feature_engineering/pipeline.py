from kedro.pipeline import Pipeline, node

from .nodes import feature_engineering

def create_pipeline(**kwargs):
    """Kedro pipeline that takes cleaned dataset as input and calls
    the feature_engineering node to generate features."""
    return Pipeline(
        [
            node(
                func=feature_engineering,
                inputs="validated_raw_data",
                outputs="engineered_data",
                name="feature_engineering_node",
            )
        ]
    )