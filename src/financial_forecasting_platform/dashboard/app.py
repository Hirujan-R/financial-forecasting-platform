import streamlit as st

from financial_forecasting_platform.dashboard.api_client import ApiClient, ApiError
from financial_forecasting_platform.dashboard.components.charts import (
    render_price_chart,
)
from financial_forecasting_platform.dashboard.components.gauges import (
    render_probability_gauge,
)
from financial_forecasting_platform.dashboard.components.market_overview import (
    render_market_overview,
)
from financial_forecasting_platform.dashboard.components.prediction_card import (
    render_prediction_card,
)
from financial_forecasting_platform.dashboard.components.shap_plot import (
    render_prediction_explanation,
    render_shap_importance,
)
from financial_forecasting_platform.dashboard.components.tables import (
    render_feature_values,
)
from financial_forecasting_platform.dashboard.styles import inject_styles
from financial_forecasting_platform.inference.load_parameters import load_parameters

TICKERS = load_parameters()["tickers"]


@st.cache_resource
def _get_client() -> ApiClient:
    return ApiClient()


def _load_prediction(ticker: str) -> dict:
    return _get_client().get_prediction(ticker)


def main() -> None:
    st.set_page_config(page_title="Volatility Regime Dashboard", layout="wide")
    inject_styles()

    st.title("Volatility Regime Prediction Dashboard")
    st.caption("Predicts whether a stock enters a high-volatility expansion regime.")

    with st.sidebar:
        st.header("Controls")
        ticker = st.selectbox("Ticker", options=TICKERS)
        predict_clicked = st.button("Predict", type="primary")

    if not predict_clicked:
        st.info("Select a ticker and press **Predict** to generate a forecast.")
        return

    try:
        prediction = _load_prediction(ticker)
    except ApiError as exc:
        st.error(str(exc))
        return

    render_prediction_card(prediction)
    render_market_overview(prediction["market"])

    col_gauge, col_shap = st.columns([1, 2])
    with col_gauge:
        render_probability_gauge(prediction["probability"])
    with col_shap:
        render_shap_importance(prediction["shap"])

    st.subheader("Price Action")
    render_price_chart(prediction["ohlcv"], prediction["bollinger"])

    col_features, col_explain = st.columns(2)
    with col_features:
        render_feature_values(prediction["features"])
    with col_explain:
        render_prediction_explanation(prediction)


if __name__ == "__main__":
    main()
