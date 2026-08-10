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
