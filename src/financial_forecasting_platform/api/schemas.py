from pydantic import BaseModel, field_validator
from typing import Literal
from financial_forecasting_platform.inference.load_parameters import load_parameters



ACCEPTED_TICKERS = load_parameters()["tickers"]


class PredictionRequest(BaseModel):

    ticker: str

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, value):

        if value not in ACCEPTED_TICKERS:
            raise ValueError(
                f"Ticker must be one of {ACCEPTED_TICKERS}"
            )

        return value


class PredictionResponse(BaseModel):
    ticker: str
    prediction: Literal[0, 1]
    probability: float

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, value):

        if value not in ACCEPTED_TICKERS:
            raise ValueError(
                f"Ticker must be one of {ACCEPTED_TICKERS}"
            )

        return value

    @field_validator("probability")
    @classmethod
    def validate_probability(cls, value):

        if not 0 <= value <= 1:
            raise ValueError(
                "Probability must be between 0 and 1"
            )

        return value
    