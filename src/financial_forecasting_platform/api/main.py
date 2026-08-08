import logging
from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException, Query

from financial_forecasting_platform.database.prediction_repository import (
    get_prediction_logs,
    insert_prediction_log,
)
from financial_forecasting_platform.inference.predictor import VolatilityPredictor

from .schemas import (
    ACCEPTED_TICKERS,
    HistoryResponse,
    PredictionRequest,
    PredictionResponse,
    RichPredictionResponse,
)

logger = logging.getLogger(__name__)

app = FastAPI()


@lru_cache
def get_predictor() -> VolatilityPredictor:
    return VolatilityPredictor()


def _ensure_valid_ticker(ticker: str) -> None:
    if ticker not in ACCEPTED_TICKERS:
        raise HTTPException(
            status_code=404,
            detail=f"Ticker must be one of {ACCEPTED_TICKERS}",
        )


def _log_prediction(result: dict) -> None:
    try:
        insert_prediction_log(
            ticker=result["ticker"],
            prediction=result["prediction"],
            probability=result["probability"],
            model_name=result["model_type"],
            model_version=result["model_version"],
        )
    except Exception:
        logger.exception(
            "Failed to persist prediction log for %s", result.get("ticker")
        )


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "model": "volatility-expansion-predictor",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(
    request: PredictionRequest,
    predictor: VolatilityPredictor = Depends(get_predictor),
):
    result = predictor.predict(request.ticker)
    _log_prediction(result)
    return result


@app.get("/prediction/{ticker}", response_model=RichPredictionResponse)
def get_prediction(
    ticker: str,
    predictor: VolatilityPredictor = Depends(get_predictor),
):
    _ensure_valid_ticker(ticker)
    result = predictor.get_prediction(ticker)
    _log_prediction(result)
    return result


@app.get("/history", response_model=list[HistoryResponse])
def history(
    limit: int = Query(default=50, ge=1, le=500),
):
    return get_prediction_logs(limit)
