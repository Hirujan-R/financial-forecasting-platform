from kedro.pipeline import Node, Pipeline  # noqa
from .nodes import outlier_detection, clip_outliers

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        Node(
                func=outlier_detection,
                inputs=["lr_X_train", "params:outlier_feature_selection"],
                outputs="outlier_detection_lr_X_train",
                name="outlier_detection_train_node"
            ),
        Node(
                func=outlier_detection,
                inputs=["lr_X_test", "params:outlier_feature_selection"],
                outputs="outlier_detection_lr_X_test",
                name="outlier_detection_test_node"
            ),
        Node(
                func=clip_outliers,
                inputs=["outlier_detection_lr_X_train", "params:clip_columns"],
                outputs="outlier_clipped_lr_X_train",
                name="winsorization_train_node"
            ),
        Node(
                func=clip_outliers,
                inputs=["outlier_detection_lr_X_test", "params:clip_columns"],
                outputs="outlier_clipped_lr_X_test",
                name="winsorization_test_node"
            )
    ])
