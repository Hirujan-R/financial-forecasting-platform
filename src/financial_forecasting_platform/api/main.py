from fastapi import FastAPI

from financial_forecasting_platform.inference.predictor import (
    VolatilityPredictor
)

from .schemas import (
    PredictionRequest,
    PredictionResponse
)


app = FastAPI()

predictor = VolatilityPredictor()

@app.get("/")
def health_check():
    return {
        "status": "ok",
        "model": "volatility-expansion-predictor"
    }




@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):

    result = predictor.predict(
        request.ticker
    )

    return result