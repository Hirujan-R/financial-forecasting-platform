from kedro.pipeline import Pipeline, node

from .nodes import feature_engineering

def create_pipeline(**kwargs):

    return Pipeline(
        [
            node(
                func=feature_engineering,
                inputs="cleaned_data",
                outputs="engineered_data",
                name="feature_engineering_node",
            )
        ]
    )