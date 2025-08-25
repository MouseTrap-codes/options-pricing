import streamlit as st

from models import black_scholes_with_greeks, dp_binomial_tree

st.set_page_config(page_title="Options Pricer", layout="centered")
st.title("Options Pricer")

st.markdown(
    """
Use this tool to price **European or American** options using either the **Black-Scholes model** (European only) or a **Binomial Tree model** (supports multiple asset types and exercise styles).
"""
)

# model selection
model = st.radio("Pricing Model", ["Black-Scholes", "Binomial Tree"])

# restrict options for Black-Scholes
if model == "Black-Scholes":
    allowed_exercise = ["european"]
    allowed_assets = ["nondividend", "dividend"]
else:
    allowed_exercise = ["european", "american"]
    allowed_assets = ["nondividend", "dividend", "currency", "future"]

# shared Inputs
col1, col2 = st.columns(2)
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

# run model
if st.button("Compute Option Price"):
    try:
        if model == "Black-Scholes":
            if asset_type not in ["nondividend", "dividend"]:
                st.warning(
                    "⚠️ Black-Scholes only supports non-dividend or dividend-paying **stocks**."
                )
            elif exercise_type != "european":
                st.warning("⚠️ Black-Scholes only supports **European-style** options.")
            else:
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

        else:
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

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
