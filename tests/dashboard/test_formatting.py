from financial_forecasting_platform.dashboard.utils.formatting import (
    direction_arrow,
    prediction_badge_class,
    prediction_label,
    prediction_short,
    regime_badge_class,
    shap_color,
)


def test_prediction_label():
    assert prediction_label(1) == "High Volatility Expansion"
    assert "Contraction" in prediction_label(0)


def test_prediction_short():
    assert prediction_short(1) == "HIGH"
    assert prediction_short(0) == "LOW"


def test_prediction_badge_class():
    assert prediction_badge_class(1) == "high"
    assert prediction_badge_class(0) == "low"


def test_shap_color():
    assert shap_color(0.5) == "#2ca02c"
    assert shap_color(-0.5) == "#d62728"
    assert shap_color(0.0) == "#2ca02c"


def test_direction_arrow():
    assert direction_arrow(0.1) == "↑"
    assert direction_arrow(-0.1) == "↓"
    assert direction_arrow(0.0) == "↑"


def test_regime_badge_class():
    assert regime_badge_class("Risk On") == "risk-on"
    assert regime_badge_class("Elevated") == "elevated"
    assert regime_badge_class("Risk Off") == "risk-off"
    assert regime_badge_class("Unknown") == "elevated"
