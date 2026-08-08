import streamlit as st

from ..styles import inject_styles
from ..utils.formatting import (
    prediction_badge_class,
    prediction_label,
    prediction_short,
)


def render_prediction_card(prediction: dict) -> None:
    inject_styles()

    probability = float(prediction["probability"])
    pred = int(prediction["prediction"])
    confidence = prediction.get("confidence", "Low")

    col_metric, col_badge, col_conf = st.columns(3)

    with col_metric:
        st.metric("Ticker", prediction["ticker"])
        close_price = float(prediction.get("close_price", 0.0))
        st.metric("Current Close", f"${close_price:,.2f}")

    with col_badge:
        st.subheader("Prediction")
        badge_class = prediction_badge_class(pred)
        st.markdown(
            f'<div class="prediction-badge {badge_class}">'
            f"{prediction_short(pred)} VOLATILITY EXPANSION</div>",
            unsafe_allow_html=True,
        )
        st.caption(prediction_label(pred))
        st.progress(probability)
        st.caption(f"Expansion probability: {probability:.0%}")

    with col_conf:
        st.metric("Probability", f"{probability:.1%}")
        st.metric("Confidence", confidence)
        st.metric("Model", prediction.get("model_type", "Unknown"))
