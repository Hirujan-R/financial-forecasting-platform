import streamlit as st


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .prediction-badge {
            padding: 0.5rem 1rem;
            border-radius: 0.5rem;
            font-weight: 700;
            text-align: center;
            color: white;
        }
        .prediction-badge.high {
            background-color: #d62728;
        }
        .prediction-badge.low {
            background-color: #2ca02c;
        }
        .regime-badge {
            padding: 0.25rem 0.75rem;
            border-radius: 0.5rem;
            font-weight: 600;
            display: inline-block;
        }
        .regime-badge.risk-on {
            background-color: #2ca02c;
            color: white;
        }
        .regime-badge.elevated {
            background-color: #ff7f0e;
            color: white;
        }
        .regime-badge.risk-off {
            background-color: #d62728;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
