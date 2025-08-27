import numpy as np
import plotly.graph_objects as go
import streamlit as st

HOVER_TEMPLATE = "Spot: %{x:$,.2f}<br>P&L: %{y:$,.2f}<extra></extra>"


def fmt_currency(v: float) -> str:
    return f"${v:,.2f}"


def compute_p_l_and_other_metrics(
    premium, K, option_type, position_type, contracts, multiplier
):
    left = 0.0 if option_type == "put" else 0.5 * K
    S = np.linspace(left, 1.5 * K, 100)
    payoff = np.zeros_like(S)
    breakeven_price = 0.0
    if option_type == "call":
        payoff = np.maximum(S - K, 0)
        breakeven_price = K + premium
    elif option_type == "put":
        payoff = np.maximum(0, K - S)
        breakeven_price = K - premium

    pnl = np.zeros_like(S)

    total_premium = premium * contracts * multiplier
    cost = 0.0
    credit = 0.0

    max_gain = 0.0
    max_loss = 0.0

    if position_type == "long":
        if option_type == "call":
            max_gain = np.inf
            max_loss = total_premium
        elif option_type == "put":
            max_gain = (K * contracts * multiplier) - total_premium
            max_loss = total_premium
        cost = total_premium
        pnl = (payoff - premium) * contracts * multiplier
    elif position_type == "short":
        if option_type == "call":
            max_gain = total_premium
            max_loss = np.inf
        elif option_type == "put":
            max_gain = total_premium
            max_loss = (K * contracts * multiplier) - total_premium
        credit = total_premium
        pnl = (premium - payoff) * contracts * multiplier

    return {
        "Profits and Losses": pnl,
        "Spot Range": S,
        "Cost": cost,
        "Credit": credit,
        "Max Gain": max_gain,
        "Max Loss": max_loss,
        "Breakeven Price": breakeven_price,
    }


# part 4: plot!
def plot_pnl(
    pnl_list, spot_range, K=None, breakeven=None, max_loss=None, max_gain=None
):
    fig = go.Figure()

    # base P&L curve
    fig.add_trace(
        go.Scatter(
            x=spot_range,
            y=pnl_list,
            mode="lines",
            name="P&L",
            line=dict(color="deepskyblue", width=2),
            hovertemplate=HOVER_TEMPLATE,
        )
    )

    # horizontal line at zero (no profit/no loss)
    fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.6)

    # strike price marker
    if K is not None:
        fig.add_vline(
            x=K,
            line_dash="dot",
            line_color="orange",
            annotation_text="Strike",
            annotation_position="bottom right",
        )

    # breakeven marker
    if breakeven is not None:
        fig.add_vline(
            x=breakeven,
            line_dash="dot",
            line_color="green",
            annotation_text="Breakeven",
            annotation_position="top right",
        )

    # max loss line if finite
    if max_loss is not None and np.isfinite(max_loss):
        fig.add_hline(
            y=-max_loss,
            line_dash="dot",
            line_color="red",
            annotation_text=f"Max Loss = ${max_loss:,.0f}",
            annotation_position="bottom right",
        )

    # max gain line if finite
    if max_gain is not None and np.isfinite(max_gain):  # ← NEW
        fig.add_hline(
            y=max_gain,
            line_dash="dot",
            line_color="lightgreen",
            annotation_text=f"Max Gain = ${max_gain:,.0f}",
            annotation_position="top right",
        )

    # shaded regions
    fig.add_trace(
        go.Scatter(
            x=spot_range,
            y=np.maximum(pnl_list, 0),
            fill="tozeroy",
            fillcolor="rgba(0,200,0,0.2)",
            line=dict(width=0),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=spot_range,
            y=np.minimum(pnl_list, 0),
            fill="tozeroy",
            fillcolor="rgba(200,0,0,0.2)",
            line=dict(width=0),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    fig.update_layout(
        title="Profit & Loss vs Spot Price",
        xaxis_title="Spot Price at Expiration ($)",
        yaxis_title="Profit / Loss ($)",
        plot_bgcolor="#262730",
        paper_bgcolor="#0D1117",
        font=dict(color="#E6E6E6", family="sans serif"),
        xaxis=dict(showgrid=True, tickprefix="$", separatethousands=True),
        yaxis=dict(showgrid=True, tickprefix="$", separatethousands=True),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)


def display_pnl(
    option_price,
    *,
    K,
    option_type,
    position_type,
    number_of_contracts,
    contract_multiplier,
):
    st.header("Profits and Losses")
    result = compute_p_l_and_other_metrics(
        K=K,
        option_type=option_type,
        position_type=position_type,
        contracts=number_of_contracts,
        multiplier=contract_multiplier,
        premium=option_price,
    )
    plot_pnl(
        result["Profits and Losses"],
        result["Spot Range"],
        K=K,
        breakeven=result["Breakeven Price"],
        max_loss=result["Max Loss"],
        max_gain=result["Max Gain"],
    )
    st.subheader("Summary Statistics")
    col1, col2, col3 = st.columns(3)

    inf = "∞"
    col1.metric("Breakeven Price", fmt_currency(result["Breakeven Price"]))
    col2.metric("Cost", fmt_currency(result["Cost"]))
    col3.metric("Credit", fmt_currency(result["Credit"]))
    col1.metric(
        "Max Gain",
        (
            inf
            if not np.isfinite(result["Max Gain"])
            else fmt_currency(result["Max Gain"])
        ),
    )
    col2.metric(
        "Max Loss",
        (
            inf
            if not np.isfinite(result["Max Loss"])
            else fmt_currency(result["Max Loss"])
        ),
    )


def plot_pnl_overlay(
    spot_range,
    *,
    pnl_model,
    pnl_market=None,
    K=None,
    breakeven_model=None,
    breakeven_market=None,
):
    fig = go.Figure()

    # model line
    fig.add_trace(
        go.Scatter(
            x=spot_range,
            y=pnl_model,
            mode="lines",
            name="Model P&L",
            line=dict(width=2),
            hovertemplate=HOVER_TEMPLATE,
        )
    )

    # market line (optional)
    if pnl_market is not None:
        fig.add_trace(
            go.Scatter(
                x=spot_range,
                y=pnl_market,
                mode="lines",
                name="Market P&L",
                line=dict(width=2, dash="dash"),
                hovertemplate=HOVER_TEMPLATE,
            )
        )

    # zero line
    fig.add_hline(y=0, line_dash="dash", opacity=0.6)

    # markers
    if K is not None:
        fig.add_vline(
            x=K,
            line_dash="dot",
            annotation_text="Strike",
            annotation_position="bottom right",
        )
    if breakeven_model is not None:
        fig.add_vline(
            x=breakeven_model,
            line_dash="dot",
            line_color="green",
            annotation_text="Model BE",
            annotation_position="top right",
        )
    if breakeven_market is not None:
        fig.add_vline(
            x=breakeven_market,
            line_dash="dot",
            line_color="lightgreen",
            annotation_text="Market BE",
            annotation_position="top left",
        )

    fig.update_layout(
        title="Profit & Loss vs Spot Price (Overlay)",
        xaxis_title="Spot Price at Expiration ($)",
        yaxis_title="Profit / Loss ($)",
        plot_bgcolor="#262730",
        paper_bgcolor="#0D1117",
        font=dict(color="#E6E6E6", family="sans serif"),
        xaxis=dict(showgrid=True, tickprefix="$", separatethousands=True),
        yaxis=dict(showgrid=True, tickprefix="$", separatethousands=True),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)


def display_pnl_overlay(
    option_price_model,
    option_price_market,
    *,
    K,
    option_type,
    position_type,
    number_of_contracts,
    contract_multiplier,
):
    st.header("Profits and Losses")
    model_result = compute_p_l_and_other_metrics(
        K=K,
        option_type=option_type,
        position_type=position_type,
        contracts=number_of_contracts,
        multiplier=contract_multiplier,
        premium=option_price_model,
    )
    market_result = compute_p_l_and_other_metrics(
        K=K,
        option_type=option_type,
        position_type=position_type,
        contracts=number_of_contracts,
        multiplier=contract_multiplier,
        premium=option_price_market,
    )
    plot_pnl_overlay(
        model_result["Spot Range"],
        pnl_model=model_result["Profits and Losses"],
        pnl_market=market_result["Profits and Losses"],
        K=K,
        breakeven_model=model_result["Breakeven Price"],
        breakeven_market=market_result["Breakeven Price"],
    )

    inf = "∞"

    st.subheader("Model Summary Statistics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Breakeven Price", fmt_currency(model_result["Breakeven Price"]))
    c2.metric("Cost", fmt_currency(model_result["Cost"]))
    c3.metric("Credit", fmt_currency(model_result["Credit"]))
    c1.metric(
        "Max Gain",
        (
            inf
            if not np.isfinite(model_result["Max Gain"])
            else fmt_currency(model_result["Max Gain"])
        ),
    )
    c2.metric(
        "Max Loss",
        (
            inf
            if not np.isfinite(model_result["Max Loss"])
            else fmt_currency(model_result["Max Loss"])
        ),
    )

    st.subheader("Market Summary Statistics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Breakeven Price", fmt_currency(market_result["Breakeven Price"]))
    c2.metric("Cost", fmt_currency(market_result["Cost"]))
    c3.metric("Credit", fmt_currency(market_result["Credit"]))
    c1.metric(
        "Max Gain",
        (
            inf
            if not np.isfinite(market_result["Max Gain"])
            else fmt_currency(market_result["Max Gain"])
        ),
    )
    c2.metric(
        "Max Loss",
        (
            inf
            if not np.isfinite(market_result["Max Loss"])
            else fmt_currency(market_result["Max Loss"])
        ),
    )
