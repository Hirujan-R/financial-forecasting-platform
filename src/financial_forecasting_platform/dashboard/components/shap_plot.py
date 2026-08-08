import plotly.graph_objects as go
import streamlit as st

from ..utils.formatting import direction_arrow, shap_color


def render_shap_importance(shap_data: list[dict]) -> None:
    if not shap_data:
        st.caption("No feature contributions available.")
        return

    contributions = [item["contribution"] for item in shap_data]
    features = [item["feature"] for item in shap_data]

    fig = go.Figure(
        go.Bar(
            x=contributions,
            y=features,
            orientation="h",
            marker=dict(
                color=[shap_color(value) for value in contributions],
            ),
            text=[f"{value:+.3f}" for value in contributions],
            textposition="outside",
        )
    )

    fig.add_vline(x=0, line_width=1, line_color="#333")

    fig.update_layout(
        title="SHAP Feature Importance",
        xaxis_title="Contribution to volatility expansion",
        height=420,
        margin=dict(l=10, r=10, t=40, b=10),
    )

    st.plotly_chart(fig, width="stretch")


def render_prediction_explanation(prediction: dict) -> None:
    probability = float(prediction.get("probability", 0.0))
    shap_data = prediction.get("shap", [])

    if not shap_data:
        st.caption("No explanation available.")
        return

    st.subheader("Prediction Explanation")
    st.markdown(f"**Probability:** {probability:.0%}")

    top_reasons = shap_data[:5]
    for item in top_reasons:
        arrow = direction_arrow(item["contribution"])
        st.markdown(
            f"- {arrow} **{item['feature']}** ({item['contribution']:+.3f})"
        )

    total = sum(abs(item["contribution"]) for item in shap_data)
    if total > 0:
        dominant = max(shap_data, key=lambda item: abs(item["contribution"]))
        direction = (
            "higher" if dominant["contribution"] >= 0 else "lower"
        )
        st.caption(
            f"Overall: conditions point toward {direction} "
            f"volatility, driven mainly by {dominant['feature']}."
        )
