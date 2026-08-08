import streamlit as st

from ..styles import inject_styles
from ..utils.formatting import regime_badge_class


def render_market_overview(market: dict) -> None:
    inject_styles()

    spy_return = market.get("spy_return")
    vix_level = market.get("vix_level")
    regime = market.get("regime", "Risk On")

    col_spy, col_vix, col_regime = st.columns(3)

    with col_spy:
        st.metric(
            "SPY Return (1d)",
            f"{spy_return:+.2%}" if spy_return is not None else "n/a",
        )

    with col_vix:
        st.metric(
            "VIX Level",
            f"{vix_level:.1f}" if vix_level is not None else "n/a",
        )

    with col_regime:
        st.subheader("Market Regime")
        badge_class = regime_badge_class(regime)
        st.markdown(
            f'<span class="regime-badge {badge_class}">{regime}</span>',
            unsafe_allow_html=True,
        )
