import pandas as pd
import streamlit as st


def render_feature_values(features: dict) -> None:
    st.subheader("Current Feature Values")

    rows = [
        {"Feature": key, "Value": value}
        for key, value in features.items()
        if isinstance(value, int | float) and value is not None
    ]

    if not rows:
        st.caption("No numeric feature values available.")
        return

    df = pd.DataFrame(rows)
    st.dataframe(df, width="stretch", hide_index=True)


def render_prediction_history(history: list[dict]) -> None:
    st.subheader("Prediction History")

    if not history:
        st.caption("No prediction history available yet.")
        return

    df = pd.DataFrame(history)

    display_cols = [
        "timestamp",
        "ticker",
        "prediction",
        "probability",
        "model_name",
        "model_version",
        "actual_outcome",
        "correct",
    ]
    df = df[[col for col in display_cols if col in df.columns]]
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M")
    df["probability"] = df["probability"].map(lambda v: f"{v:.1%}")

    st.dataframe(df, width="stretch", hide_index=True)


def render_model_stats(model_stats: dict) -> None:
    st.subheader("Model Statistics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        value = model_stats.get("accuracy")
        st.metric("Accuracy", f"{value:.1%}" if value is not None else "n/a")
    with col2:
        value = model_stats.get("precision")
        st.metric("Precision", f"{value:.1%}" if value is not None else "n/a")
    with col3:
        value = model_stats.get("recall")
        st.metric("Recall", f"{value:.1%}" if value is not None else "n/a")
    with col4:
        value = model_stats.get("roc_auc")
        st.metric("ROC-AUC", f"{value:.3f}" if value is not None else "n/a")
