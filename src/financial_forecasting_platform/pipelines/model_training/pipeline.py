from kedro.pipeline import Node, Pipeline  # noqa
from .nodes import create_lr_pipeline, create_xgb_pipeline, train_model
from financial_forecasting_platform.utils.mlflow_utils import log_sklearn_model

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
                inputs=["outlier_clipped_lr_X_train", "lr_y_train",
                        "lr_training_pipeline", "params:lr_param_grid"],
                outputs={
                    "model": "lr_model",
                    "params": "lr_params"
                },
                name="train_lr_model_node"
            ),
        Node(
                func=log_sklearn_model,
                inputs=["lr_model", "lr_params", "params:lr_training_run_name"],
                outputs="lr_mlflow_model_uri",
                name="log_lr_model_node"
            ),
        Node(
                func=create_xgb_pipeline,
                inputs=["params:onehot_chr_features", "params:ordinal_chr_features"],
                outputs="xgb_training_pipeline",
                name="create_xgb_pipeline_node"
            ),
        Node(
                func=train_model,
                inputs=["xgb_X_train", "xgb_y_train",
                        "xgb_training_pipeline", "params:xgb_param_grid"],
                outputs={
                    "model": "xgb_model",
                    "params": "xgb_params"
                },
                name="train_xgb_model_node"
            ),
        Node(
                func=log_sklearn_model,
                inputs=["xgb_model", "xgb_params", "params:xgb_training_run_name"],
                outputs="xgb_mlflow_model_uri",
                name="log_xgb_model_node"
            )
        
    ])
