from kedro.pipeline import Node, Pipeline  # noqa
from .nodes import create_lr_pipeline, create_xgb_pipeline, train_model

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        Node(
                func=create_lr_pipeline,
                inputs=["outlier_clipped_lr_X_train", "params:onehot_chr_features", "params:ordinal_chr_features", 
                        "params:skewed_features"],
                outputs="lr_training_pipeline",
                name="create_lr_pipeline_node"
            ),
        Node(
                func=train_model,
                inputs=["params:lr_experiment_tags", "outlier_clipped_lr_X_train", "lr_y_train",
                        "outlier_clipped_lr_X_test", "lr_y_test",
                        "lr_training_pipeline", "params:lr_param_grid"],
                outputs="lr_mlflow_model_uri",
                name="train_lr_model_node"
            ),
        Node(
                func=create_xgb_pipeline,
                inputs=["params:onehot_chr_features", "params:ordinal_chr_features"],
                outputs="xgb_training_pipeline",
                name="create_xgb_pipeline_node"
            ),
        Node(
                func=train_model,
                inputs=["params:xgb_experiment_tags","xgb_X_train", "xgb_y_train",
                        "xgb_X_test", "xgb_y_test",
                        "xgb_training_pipeline", "params:xgb_param_grid"],
                outputs="xgb_mlflow_model_uri",
                name="train_xgb_model_node"
            ) 
    ])
