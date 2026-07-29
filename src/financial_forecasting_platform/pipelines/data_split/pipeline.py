from kedro.pipeline import Node, Pipeline  
from .nodes import split_features_target, split_train_test


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        Node(
                func=split_features_target,
                inputs=["lr_features", "params:target_variable"],
                outputs={
                    "X": "lr_X",
                    "y": "lr_y"
                },
                name="split_lr_features_target_node"
            ),
        Node(
                func=split_train_test,
                inputs=["lr_X", "lr_y", "params:training_proportion"],
                outputs={
                    "X_train": "lr_X_train",
                    "X_test": "lr_X_test",
                    "y_train": "lr_y_train",
                    "y_test": "lr_y_test"
                },
                name="split_lr_train_test_node"
            ),
        Node(
                func=split_features_target,
                inputs=["xgboost_features", "params:target_variable"],
                outputs={
                    "X": "xgb_X",
                    "y": "xgb_y"
                },
                name="split_xgb_features_target_node"
            ),
        Node(
                func=split_train_test,
                inputs=["xgb_X", "xgb_y", "params:training_proportion"],
                outputs={
                    "X_train": "xgb_X_train",
                    "X_test": "xgb_X_test",
                    "y_train": "xgb_y_train",
                    "y_test": "xgb_y_test"
                },
                name="split_xgb_train_test_node"
            )
    ])
