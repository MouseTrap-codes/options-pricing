from datetime import datetime

import streamlit as st
import yfinance as yf

from models import black_scholes_with_greeks, dp_binomial_tree
from pnl_helpers import display_pnl

st.set_page_config(page_title="Options Pricer", layout="centered")
st.title("Options Pricer")

st.markdown(
    """
Use this tool to price **European or American** options using either the **Black-Scholes model** (European only) or a **Binomial Tree model** (supports multiple asset types and exercise styles).
"""
)

# part 1: inputs
# model selection
model = st.radio("Pricing Model", ["Black-Scholes", "Binomial Tree"])

allowed_positions = ["long", "short"]

# restrict options for Black-Scholes
if model == "Black-Scholes":
    allowed_exercise = ["european"]
    allowed_assets = ["nondividend", "dividend"]
else:
    allowed_exercise = ["european", "american"]
    allowed_assets = ["nondividend", "dividend", "currency", "future"]

# part 0: live option chains
st.subheader("Live Option Chain Data (Optional)")
use_live = st.checkbox("Use Live Market Data")

# shared Inputs
col1, col2, col3 = st.columns(3)
with col1:
    S_t = st.number_input("Current Stock Price (S₀)", value=100.0, step=1.0)
    K = st.number_input("Strike Price (K)", value=100.0, step=1.0)
    T = st.number_input("Time to Maturity (T, years)", value=1.0, step=0.1)
    r = st.number_input("Risk-Free Rate (r)", value=0.05, step=0.01)
    sigma = st.number_input("Volatility (σ)", value=0.2, step=0.01)
with col2:
    option_type = st.selectbox("Option Type", ["call", "put"])
    exercise_type = st.selectbox("Exercise Style", allowed_exercise)
    asset_type = st.selectbox("Asset Type", allowed_assets)
    position_type = st.selectbox("Position Type", allowed_positions)
with col3:
    number_of_contracts = st.number_input(
        "Number of Contracts", min_value=1, value=1, step=1
    )
    contract_multiplier = st.number_input(
        "Contract Multiplier", min_value=1, value=100, step=1
    )

if use_live:
    ticker = st.text_input("Enter Stock Ticker (i.e. AAPL)", value="AAPL")
    if ticker:
        try:
            stock = yf.Ticker(ticker)
            S_t = stock.history(period="1d")["Close"].iloc[-1]

            # show expiration dates
            try:
                expirations = stock.options
            except Exception as e:
                st.error(f"Could not fetch option expirations: {e}")
                expirations = []
            expiry_str = str(
                st.selectbox(
                    "Choose Expiration Date",
                    expirations,
                    index=0 if expirations else None,
                )
            )
            expiry_date = datetime.strptime(expiry_str, "%Y-%m-%d")
            T = (expiry_date - datetime.now()).days / 365

            # fetch option chain
            chain = stock.option_chain(expiry_str)
            df = (
                chain.calls
                if st.radio("Option Type", ["call", "put"]) == "call"
                else chain.puts
            )

            K = st.selectbox("Select Strike Price (K)", sorted(df["strike"].tolist()))
        except Exception as e:
            st.error(f"Error fetching data for {ticker}: {e}")

sigma = st.number_input("Volatility (σ)", value=0.2, step=0.01)
r = st.number_input("Risk-Free Rate (r)", value=0.05, step=0.01)

number_of_contracts = st.number_input(
    "Number of Contracts", min_value=1, value=1, step=1
)
contract_multiplier = st.number_input(
    "Contract Multiplier", min_value=1, value=100, step=1
)

# additional fields depending on asset type
q = r_f = None
if asset_type == "dividend":
    q = st.number_input("Dividend Yield (q)", value=0.02, step=0.005)
elif asset_type == "currency":
    r_f = st.number_input("Foreign Interest Rate (r_f)", value=0.01, step=0.005)

# bt-specific inputs
if model == "Binomial Tree":
    N = st.slider(
        "Number of Time Steps (N)", min_value=10, max_value=1000, value=200, step=10
    )
else:
    N = None

# part 2: compute premium
# run model
if st.button("Compute Option Price"):
    try:
        if model == "Black-Scholes":
            if asset_type not in ["nondividend", "dividend"]:
                st.warning(
                    "Black-Scholes only supports non-dividend or dividend-paying **stocks**."
                )
            elif exercise_type != "european":
                st.warning("Black-Scholes only supports **European-style** options.")
            else:
                K = float(K) if K is not None else 100.0  # fallback default
                result = black_scholes_with_greeks(
                    S_t=S_t,
                    K=K,
                    T=T,
                    r=r,
                    sigma=sigma,
                    q=q or 0.0,
                    option_type=option_type,
                )
                st.success(f"Option Price: **{result['Price']:.4f}**")
                st.subheader("Greeks")
                st.table({k: [v] for k, v in result.items() if k != "Price"})
                display_pnl(
                    option_price=result["Price"],
                    K=K,
                    option_type=option_type,
                    position_type=position_type,
                    number_of_contracts=number_of_contracts,
                    contract_multiplier=contract_multiplier,
                )
        else:
            K = float(K) if K is not None else 100.0  # fallback default
            price = dp_binomial_tree(
                S_t=S_t,
                K=K,
                T=T,
                r=r,
                sigma=sigma,
                option_type=option_type,
                exercise_type=exercise_type,
                asset_type=asset_type,
                N=N,  # type: ignore
                q=q,
                r_f=r_f,
            )
            st.success(f"Option Price: **{price:.4f}**")
            display_pnl(
                option_price=price,
                K=K,
                option_type=option_type,
                position_type=position_type,
                number_of_contracts=number_of_contracts,
                contract_multiplier=contract_multiplier,
            )

    except Exception as e:
        st.error(f"Error: {e}")
