import mlflow
import mlflow.sklearn

def log_sklearn_model(
    model,
    params,
    model_name: str):
    """
    Logs sklearn compatible models to MLflow.
    """
    mlflow.log_params(params)
    
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        name=model_name,
        registered_model_name="market_direction_model",
        skops_trusted_types=[
            "xgboost.core.Booster",
            "xgboost.sklearn.XGBClassifier"
        ]
    )

    return model_info.model_uri