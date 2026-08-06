import mlflow
import mlflow.sklearn


def load_champion_model():
    model = mlflow.sklearn.load_model(
        "models:/volatility-expansion-predictor@champion"
    )
    return model

