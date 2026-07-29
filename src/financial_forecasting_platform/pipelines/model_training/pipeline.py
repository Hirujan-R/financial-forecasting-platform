from kedro.pipeline import Node, Pipeline  # noqa
from .nodes import create_lr_pipeline, train_lr_model
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
                func=train_lr_model,
                inputs=["outlier_clipped_lr_X_train", "lr_y_train",
                        "lr_training_pipeline", "params:lr_param_grid"],
                outputs={
                    "model": "model",
                    "params": "params"
                },
                name="train_lr_model_node"
            ),
        Node(
                func=log_sklearn_model,
                inputs=["model", "params", "params:lr_training_run_name"],
                outputs="lr_mlflow_model_uri",
                name="log_lr_model_node"
            )
        
    ])
