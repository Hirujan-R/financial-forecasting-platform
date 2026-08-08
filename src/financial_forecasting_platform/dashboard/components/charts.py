import plotly.graph_objects as go
import streamlit as st


def render_price_chart(ohlcv: list[dict], bollinger: dict | None = None) -> None:
    dates = [point["date"] for point in ohlcv]
    volume = [point["volume"] for point in ohlcv]

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=dates,
            open=[point["open"] for point in ohlcv],
            high=[point["high"] for point in ohlcv],
            low=[point["low"] for point in ohlcv],
            close=[point["close"] for point in ohlcv],
            name="OHLC",
            yaxis="y",
        )
    )

    if bollinger:
        fig.add_trace(
            go.Scatter(
                x=bollinger["dates"],
                y=bollinger["upper"],
                mode="lines",
                name="Bollinger Upper",
                line=dict(color="#d62728", width=1),
                yaxis="y",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=bollinger["dates"],
                y=bollinger["middle"],
                mode="lines",
                name="Bollinger Middle",
                line=dict(color="#7f7f7f", width=1),
                yaxis="y",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=bollinger["dates"],
                y=bollinger["lower"],
                mode="lines",
                name="Bollinger Lower",
                line=dict(color="#2ca02c", width=1),
                fill="tonexty",
                yaxis="y",
            )
        )

    fig.add_trace(
        go.Bar(
            x=dates,
            y=volume,
            name="Volume",
            yaxis="y2",
            marker=dict(color="rgba(150, 150, 150, 0.4)"),
        )
    )

    fig.update_layout(
        xaxis=dict(
            rangeslider=dict(visible=False),
            rangeselector=dict(
                buttons=[
                    dict(count=30, label="1M", step="day", stepmode="backward"),
                    dict(count=90, label="3M", step="day", stepmode="backward"),
                    dict(count=180, label="6M", step="day", stepmode="backward"),
                    dict(step="all", label="All"),
                ]
            ),
        ),
        yaxis=dict(title="Price"),
        yaxis2=dict(title="Volume", overlaying="y", side="right", showgrid=False),
        height=480,
        margin=dict(l=60, r=60, t=30, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )

    st.plotly_chart(fig, width="stretch")
