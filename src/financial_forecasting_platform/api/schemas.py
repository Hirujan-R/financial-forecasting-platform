from datetime import datetime
from typing import Literal

from pydantic import BaseModel, field_validator

from financial_forecasting_platform.inference.load_parameters import load_parameters

ACCEPTED_TICKERS = load_parameters()["tickers"]


def _validate_ticker(value: str) -> str:
    if value not in ACCEPTED_TICKERS:
        raise ValueError(
            f"Ticker must be one of {ACCEPTED_TICKERS}"
        )
    return value


class PredictionRequest(BaseModel):

    ticker: str

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, value):
        return _validate_ticker(value)


class PredictionResponse(BaseModel):
    ticker: str
    prediction: Literal[0, 1]
    probability: float
    confidence: str
    close_price: float
    model_type: str
    model_version: int | str

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, value):
        return _validate_ticker(value)

    @field_validator("probability")
    @classmethod
    def validate_probability(cls, value):
        if not 0 <= value <= 1:
            raise ValueError(
                "Probability must be between 0 and 1"
            )
        return value


class OhlcvPoint(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int


class BollingerData(BaseModel):
    dates: list[str]
    upper: list[float | None]
    middle: list[float | None]
    lower: list[float | None]


class ShapContribution(BaseModel):
    feature: str
    contribution: float


class MarketOverview(BaseModel):
    spy_return: float | None
    vix_level: float | None
    regime: str


class ModelStats(BaseModel):
    accuracy: float | None
    precision: float | None
    recall: float | None
    roc_auc: float | None


class RichPredictionResponse(PredictionResponse):
    model_stats: ModelStats
    ohlcv: list[OhlcvPoint]
    bollinger: BollingerData
    features: dict[str, float | str | None]
    shap: list[ShapContribution]
    market: MarketOverview


class HistoryResponse(BaseModel):
    prediction_id: str
    timestamp: datetime
    ticker: str
    prediction: int
    probability: float
    model_name: str
    model_version: int | str
    actual_outcome: int | None
    correct: bool | None
