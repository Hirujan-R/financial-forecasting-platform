def prediction_label(prediction: int) -> str:
    return "High Volatility Expansion" if prediction == 1 else "Low Volatility / Contraction"


def prediction_short(prediction: int) -> str:
    return "HIGH" if prediction == 1 else "LOW"


def prediction_badge_class(prediction: int) -> str:
    return "high" if prediction == 1 else "low"


def confidence_color(confidence: str) -> str:
    return {
        "High": "normal",
        "Medium": "inverse",
        "Low": "off",
    }.get(confidence, "off")


def shap_color(contribution: float) -> str:
    return "#2ca02c" if contribution >= 0 else "#d62728"


def direction_arrow(contribution: float) -> str:
    return "↑" if contribution >= 0 else "↓"


def regime_badge_class(regime: str) -> str:
    mapping = {
        "Risk On": "risk-on",
        "Elevated": "elevated",
        "Risk Off": "risk-off",
    }
    return mapping.get(regime, "elevated")
