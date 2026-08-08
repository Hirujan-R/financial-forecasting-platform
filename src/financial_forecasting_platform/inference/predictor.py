import numpy as np
import pandas as pd
import scipy.sparse as sp
import shap
from mlflow import MlflowClient

from .engineer_data import engineer_data
from .get_market_data import get_market_data
from .load_model import load_champion_model

HIGH_CONFIDENCE_THRESHOLD = 0.75
MEDIUM_CONFIDENCE_THRESHOLD = 0.6
RISK_OFF_VIX_LEVEL = 25.0
ELEVATED_VIX_LEVEL = 18.0
EXPANSION_CLASS_INDEX = 1


def _json_safe(value):
    if value is None:
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float) and pd.isna(value):
        return None
    return value


def _latest_non_null(series: pd.Series):
    non_null = series.dropna()
    if non_null.empty:
        return None
    return _json_safe(non_null.iloc[-1])


def _compute_bollinger(
    df: pd.DataFrame,
    window_size: int = 20,
    num_std: float = 2.0,
) -> dict:
    """Computes Bollinger bands aligned to the rows of ``df``."""
    close = df["Close"]
    middle = close.rolling(window_size).mean()
    std = close.rolling(window_size).std()

    return {
        "dates": [pd.Timestamp(date).isoformat() for date in df["Date"]],
        "upper": [_json_safe(v) for v in (middle + num_std * std)],
        "middle": [_json_safe(v) for v in middle],
        "lower": [_json_safe(v) for v in (middle - num_std * std)],
    }


def _ohlcv_records(df: pd.DataFrame) -> list[dict]:
    records = []
    for row in df.itertuples():
        records.append(
            {
                "date": pd.Timestamp(row.Date).isoformat(),
                "open": float(row.Open),
                "high": float(row.High),
                "low": float(row.Low),
                "close": float(row.Close),
                "volume": int(row.Volume),
            }
        )
    return records


class VolatilityPredictor:

    def __init__(self):
        self.model = load_champion_model()
        self.client = MlflowClient()
        self.model_version = self.client.get_model_version_by_alias(
            "volatility-expansion-predictor",
            "champion"
        )
        self.model_type = self.model_version.tags["model"]
        self.model_stats = self._load_model_stats()

    def _prepare_features(
        self,
        ticker: str,
        lookback_window: int = 500,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        stock_data, spy_data, vix_data = get_market_data(ticker, lookback_window)
        feature_engineered_data = engineer_data(
            stock_data,
            spy_data,
            vix_data,
            self.model_type
        )
        return stock_data, feature_engineered_data

    def _load_model_stats(self) -> dict:
        run = self.client.get_run(self.model_version.run_id)
        metrics = run.data.metrics

        return {
            "accuracy": _json_safe(metrics.get("accuracy")),
            "precision": _json_safe(metrics.get("precision")),
            "recall": _json_safe(metrics.get("recall")),
            "roc_auc": _json_safe(metrics.get("roc_auc")),
        }

    @staticmethod
    def _confidence(probability: float) -> str:
        if probability >= HIGH_CONFIDENCE_THRESHOLD:
            return "High"
        if probability >= MEDIUM_CONFIDENCE_THRESHOLD:
            return "Medium"
        return "Low"

    def _score(self, feature_engineered_data: pd.DataFrame) -> tuple[int, float]:
        latest_features = feature_engineered_data.tail(1)
        prediction = self.model.predict(latest_features)
        probability = self.model.predict_proba(latest_features)[:, 1]

        return int(prediction[0]), float(probability[0])

    def predict(
        self,
        ticker: str,
        lookback_window: int = 500,
    ) -> dict:
        stock_data, feature_engineered_data = self._prepare_features(
            ticker,
            lookback_window
        )
        prediction, probability = self._score(feature_engineered_data)

        return {
            "ticker": ticker,
            "prediction": prediction,
            "probability": probability,
            "confidence": self._confidence(probability),
            "close_price": _json_safe(stock_data["Close"].iloc[-1]),
            "model_type": self.model_type,
            "model_version": _json_safe(self.model_version.version),
        }

    def get_latest_features(
        self,
        ticker: str,
        lookback_window: int = 500,
    ) -> dict:
        stock_data, feature_engineered_data = self._prepare_features(
            ticker,
            lookback_window
        )
        latest = feature_engineered_data.tail(1)

        return {
            "ohlcv": _ohlcv_records(stock_data),
            "bollinger": _compute_bollinger(stock_data),
            "features": {
                _json_safe(key): _json_safe(value)
                for key, value in latest.to_dict(orient="records")[0].items()
            },
        }

    def _explain_features(
        self,
        feature_engineered_data: pd.DataFrame,
        top_n: int = 10,
        background_size: int = 150,
    ) -> list[dict]:
        latest = feature_engineered_data.tail(1)

        background = feature_engineered_data.dropna().iloc[-background_size:]
        if background.empty:
            background = feature_engineered_data.iloc[-background_size:]

        preprocessor = self.model.named_steps["preprocessor"]
        classifier_key = "xgb" if self.model_type == "XGBoost" else "logreg"
        classifier = self.model.named_steps[classifier_key]

        X_background = preprocessor.transform(background)
        X_row = preprocessor.transform(latest)

        if sp.issparse(X_background):
            X_background = X_background.toarray()
        if sp.issparse(X_row):
            X_row = X_row.toarray()
        if X_background.dtype == object:
            X_background = X_background.astype(float)
        if X_row.dtype == object:
            X_row = X_row.astype(float)

        if self.model_type == "XGBoost":
            explainer = shap.TreeExplainer(classifier, data=X_background)
        else:
            explainer = shap.LinearExplainer(
                classifier,
                masker=shap.maskers.Independent(X_background),
            )

        shap_values = explainer.shap_values(X_row)

        if isinstance(shap_values, list):
            shap_values = shap_values[EXPANSION_CLASS_INDEX]
        if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:  # noqa: PLR2004
            shap_values = shap_values[:, :, EXPANSION_CLASS_INDEX]
        if isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:  # noqa: PLR2004
            if shap_values.shape[0] == 1:
                shap_values = shap_values[0]  # noqa: PLR2004

        contributions = np.asarray(shap_values).flatten()
        feature_names = preprocessor.get_feature_names_out()

        importance = [
            {"feature": str(name), "contribution": float(contrib)}
            for name, contrib in zip(feature_names, contributions)
        ]
        importance.sort(key=lambda item: abs(item["contribution"]), reverse=True)

        return importance[:top_n]

    def _market_overview(self, feature_engineered_data: pd.DataFrame) -> dict:
        spy_return = _latest_non_null(
            feature_engineered_data.get("spy_lag_return_1", pd.Series(dtype=float))
        )
        vix_level = _latest_non_null(
            feature_engineered_data.get("vix_level", pd.Series(dtype=float))
        )

        if vix_level is not None and vix_level >= RISK_OFF_VIX_LEVEL:
            regime = "Risk Off"
        elif vix_level is not None and vix_level >= ELEVATED_VIX_LEVEL:
            regime = "Elevated"
        else:
            regime = "Risk On"

        return {
            "spy_return": spy_return,
            "vix_level": vix_level,
            "regime": regime,
        }

    def explain(
        self,
        ticker: str,
        lookback_window: int = 500,
        top_n: int = 10,
    ) -> list[dict]:
        _, feature_engineered_data = self._prepare_features(
            ticker,
            lookback_window
        )
        return self._explain_features(
            feature_engineered_data,
            top_n=top_n,
        )

    def get_prediction(
        self,
        ticker: str,
        lookback_window: int = 500,
        top_n: int = 10,
        background_size: int = 150,
    ) -> dict:
        stock_data, feature_engineered_data = self._prepare_features(
            ticker,
            lookback_window
        )
        prediction, probability = self._score(feature_engineered_data)

        latest = feature_engineered_data.tail(1)
        shap_importance = self._explain_features(
            feature_engineered_data,
            top_n=top_n,
            background_size=background_size,
        )

        return {
            "ticker": ticker,
            "prediction": prediction,
            "probability": probability,
            "confidence": self._confidence(probability),
            "close_price": _json_safe(stock_data["Close"].iloc[-1]),
            "model_type": self.model_type,
            "model_version": _json_safe(self.model_version.version),
            "model_stats": self.model_stats,
            "ohlcv": _ohlcv_records(stock_data),
            "bollinger": _compute_bollinger(stock_data),
            "features": {
                _json_safe(key): _json_safe(value)
                for key, value in latest.to_dict(orient="records")[0].items()
            },
            "shap": shap_importance,
            "market": self._market_overview(feature_engineered_data),
        }

