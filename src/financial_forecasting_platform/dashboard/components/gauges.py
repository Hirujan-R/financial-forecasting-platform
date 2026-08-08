import plotly.graph_objects as go
import streamlit as st

from ..utils.formatting import confidence_color


def render_probability_gauge(probability: float) -> None:
    color = "#d62728" if probability >= 0.5 else "#2ca02c"  # noqa: PLR2004

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "font": {"size": 44}},
            title={"text": "Volatility Expansion Probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 50], "color": "#2ca02c"},
                    {"range": [50, 100], "color": "#d62728"},
                ],
                "threshold": {
                    "line": {"color": "#333", "width": 3},
                    "thickness": 0.8,
                    "value": 50,
                },
            },
        )
    )
    fig.update_layout(height=260, margin=dict(l=30, r=30, t=60, b=10))
    st.plotly_chart(fig, width="stretch")


def render_confidence_gauge(confidence: str) -> None:
    value = {"High": 0.9, "Medium": 0.65, "Low": 0.3}.get(confidence, 0.3)
    color = {"High": "#2ca02c", "Medium": "#ff7f0e", "Low": "#d62728"}.get(
        confidence, "#d62728"
    )

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": f"Confidence: {confidence}"},
            gauge={
                "axis": {"range": [0, 1]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 0.4], "color": "#ef8a62"},
                    {"range": [0.4, 0.75], "color": "#fdcc8a"},
                    {"range": [0.75, 1], "color": "#a6d96a"},
                ],
            },
        )
    )
    fig.update_layout(height=200, margin=dict(l=30, r=30, t=60, b=10))
    st.plotly_chart(fig, width="stretch")


def render_confidence_metric(confidence: str) -> None:
    st.metric(
        "Confidence",
        confidence,
        delta=None,
        delta_color=confidence_color(confidence),
    )
