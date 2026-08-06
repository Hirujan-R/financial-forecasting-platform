import pandas as pd
from mlflow import MlflowClient
from .get_market_data import get_market_data
from .load_model import load_champion_model
from .engineer_data import engineer_data


class VolatilityPredictor:

    def __init__(self):
        self.model = load_champion_model()
        self.client = MlflowClient()
        self.model_version = self.client.get_model_version_by_alias(
            "volatility-expansion-predictor",
            "champion"
        )
        self.model_type = self.model_version.tags["model"]


    def predict(self, ticker: str):
        
        stock_data, spy_data, vix_data = get_market_data(ticker, 500)
        feature_engineered_data = engineer_data(
            stock_data,
            spy_data,
            vix_data,
            self.model_type)
        
        latest_features = feature_engineered_data.tail(1)

        prediction = self.model.predict(latest_features)

        probability = self.model.predict_proba(latest_features)[:, 1]

        return {
            "ticker": ticker,
            "prediction": int(prediction[0]),
            "probability": float(probability[0])
        }
    

if __name__ == "__main__":
    vp = VolatilityPredictor()
    print(vp.predict("GOOG"))