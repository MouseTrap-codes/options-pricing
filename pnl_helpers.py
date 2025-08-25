import numpy as np
import plotly.graph_objects as go
import streamlit as st


def compute_p_l_and_other_metrics(
    premium, K, option_type, position_type, contracts, multiplier
):
    S = np.linspace(0.5 * K, 1.5 * K, 100)
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
def plot_pnl(pnl_list, spot_range, K=None, breakeven=None, max_loss=None):
    fig = go.Figure()

    # base P&L curve
    fig.add_trace(
        go.Scatter(
            x=spot_range,
            y=pnl_list,
            mode="lines",
            name="P&L",
            line=dict(color="deepskyblue", width=2),
            hovertemplate="Spot: %{x}<br>P&L: %{y}",
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

    # max loss line (only if finite)
    if max_loss is not None and np.isfinite(max_loss):
        fig.add_hline(
            y=-max_loss,
            line_dash="dot",
            line_color="red",
            annotation_text=f"Max Loss = {-max_loss:.0f}",
            annotation_position="bottom right",
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
        xaxis_title="Spot Price at Expiration (S)",
        yaxis_title="Profit / Loss",
        plot_bgcolor="#262730",
        paper_bgcolor="#0D1117",
        font=dict(color="#E6E6E6", family="sans serif"),
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True),
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
    )
    st.subheader("Summary Statistics")
    col1, col2, col3 = st.columns(3)

    col1.metric("Breakeven Price", f"{result['Breakeven Price']:.2f}")
    col2.metric("Cost", f"{result['Cost']:.2f}")
    col3.metric("Credit", f"{result['Credit']:.2f}")

    col1.metric(
        "Max Gain",
        "∞" if not np.isfinite(result["Max Gain"]) else f"{result['Max Gain']:.2f}",
    )
    col2.metric(
        "Max Loss",
        "∞" if not np.isfinite(result["Max Loss"]) else f"{result['Max Loss']:.2f}",
    )
