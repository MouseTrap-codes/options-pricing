from datetime import datetime
from typing import Optional, cast

import numpy as np
import pandas as pd
import streamlit as st

from models import black_scholes_with_greeks, dp_binomial_tree
from option_chain_helpers import (
    _expirations,
    _last_close,
    _market_premium_for_strike,
    _option_chain,
    _yearfrac,
    compute_position_greeks,
    implied_volatility,
)
from pnl_helpers import display_pnl, display_pnl_overlay

st.set_page_config(page_title="Options Pricer + P&L", layout="centered")
st.title("Options Pricer + P&L")

st.markdown(
    """
Use this tool to price **European or American** options using either the **Black-Scholes model** (European only & nondividend/divdend only) or a **Binomial Tree model** (supports multiple asset types i.e. currency and exercise styles i.e. American).
"""
)

use_live = st.checkbox("Use Live Market Data", value=True)
ticker = None
if use_live:
    ticker = st.text_input("Stock Ticker", value="AAPL")

# part 1: inputs
st.subheader("Inputs")
model = st.radio("Pricing Model", ["Black-Scholes", "Binomial Tree"])
allowed_positions = ["long", "short"]

if model == "Black-Scholes":
    allowed_exercise = ["european"]
    allowed_assets = ["nondividend", "dividend"]
else:
    allowed_exercise = ["european", "american"]
    allowed_assets = ["nondividend", "dividend", "currency", "future"]

# Make K always defined; override later depending on mode
K: float = 100.0

# shared Inputs
col1, col2, col3 = st.columns(3)
with col1:
    S_t = st.number_input(
        "Current Stock Price (S₀)", value=100.0, step=1.0, disabled=use_live
    )
    # Show numeric K only when NOT using live (so we never show two K inputs)
    if not use_live:
        K = st.number_input("Strike Price (K)", value=100.0, step=1.0, key="K_manual")
    T = st.number_input(
        "Time to Maturity (T, years)", value=1.0, step=0.1, disabled=use_live
    )
    r = st.number_input("Risk-Free Rate (r)", value=0.05, step=0.01)
    sigma = st.number_input("Volatility (σ)", min_value=0.00, value=0.2, step=0.01)

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

# additional fields depending on asset type
q = r_f = None
if asset_type == "dividend":
    q = st.number_input("Dividend Yield (q)", value=0.02, step=0.005)
elif asset_type == "currency":
    r_f = st.number_input("Foreign Interest Rate (r_f)", value=0.01, step=0.005)

# bt-specific inputs
N: Optional[int] = None
if model == "Binomial Tree":
    N = st.slider(
        "Number of Time Steps (N)", min_value=10, max_value=1000, value=200, step=10
    )
else:
    N = None

EMPTY_CHAIN_DF: pd.DataFrame = pd.DataFrame(
    {
        "strike": pd.Series(dtype="float64"),
        "bid": pd.Series(dtype="float64"),
        "ask": pd.Series(dtype="float64"),
        "lastPrice": pd.Series(dtype="float64"),
    }
)

df_chain: pd.DataFrame = EMPTY_CHAIN_DF.copy()
sigma_eff = float(sigma)
# live wiring: S₀, expiry->T, and K from chain (ATM default)
if use_live and ticker:
    try:
        with st.spinner("Fetching live data…"):
            S_t = _last_close(ticker)  # live spot
            expirations = _expirations(ticker)

        if expirations:
            # metrics row *above* the expiry dropdown
            m1, m2 = st.columns(2)
            m1.metric("Live S₀", f"{S_t:.2f}")
            t_placeholder = m2.empty()

            # expiry dropdown
            expiry_str = cast(
                str, st.selectbox("Choose Expiration Date", expirations, index=0)
            )
            expiry_date = datetime.strptime(expiry_str, "%Y-%m-%d")
            T = _yearfrac(expiry_date)  # override T
            t_placeholder.metric("T (years)", f"{T:.6f}")

            # option chain + single K input (selectbox)
            with st.spinner("Loading option chain..."):
                oc = _option_chain(ticker, expiry_str)
            df = oc.calls if option_type == "call" else oc.puts
            df_chain = df

            if df is not None and not df.empty and "strike" in df.columns:
                strikes = sorted(
                    float(x) for x in df["strike"].dropna().unique().tolist()
                )
                if strikes:
                    spot_for_atm = float(S_t)
                    atm_idx = min(
                        range(len(strikes)),
                        key=lambda i: abs(strikes[i] - spot_for_atm),
                    )
                    K = st.selectbox(
                        "Strike Price (K)", strikes, index=atm_idx, key="K_live"
                    )
                else:
                    st.warning("No strikes found; enter strike manually below.")
                    K = st.number_input(
                        "Strike Price (K)", value=100.0, step=1.0, key="K_live_manual"
                    )
            else:
                st.warning("Empty option chain; enter strike manually below.")
                K = st.number_input(
                    "Strike Price (K)", value=100.0, step=1.0, key="K_live_manual"
                )

            st.caption(
                f"Using live S₀ = {S_t:.2f} • K = {float(K):.2f} • T = {T:.6f} years"
            )

            # precompute market mid and IV
            market_premium = None
            iv_market = None
            use_market_iv = False
            sigma_eff = float(sigma)

            if use_live and ticker and (df_chain is not None) and not df_chain.empty:
                try:
                    market_premium = _market_premium_for_strike(df=df_chain, K=K)
                except Exception as e:
                    st.warning(f"Could not get market premium for K={K}: {e}")

                if market_premium is not None:
                    # precompute with correct model
                    if model == "Black-Scholes":

                        def price_of_sigma(sig: float) -> float:
                            return black_scholes_with_greeks(
                                S_t=S_t,
                                K=K,
                                T=T,
                                r=r,
                                sigma=sig,
                                q=(q or 0.0),
                                option_type=option_type,
                            )["Price"]

                    else:
                        N_int = cast(int, N)
                        if (N is None) or (T is None) or (T <= 0):
                            sigma_floor = 1e-4
                            N_int = 200  # a safe fallback for IV solver callback
                        else:
                            N_int = int(N)
                            dt = T / N_int
                            # drift b depends on asset type
                            if asset_type == "currency":
                                b = r - (r_f or 0.0)
                            elif asset_type == "future":
                                b = 0.0
                            else:
                                b = r - (q or 0.0)
                            sigma_floor = max(
                                1e-4, 1.05 * abs(b) * np.sqrt(dt)
                            )  # 5% buffer

                        def price_of_sigma(sig: float) -> float:
                            try:
                                return dp_binomial_tree(
                                    S_t=S_t,
                                    K=K,
                                    T=T,
                                    r=r,
                                    sigma=sig,
                                    option_type=option_type,
                                    exercise_type=exercise_type,
                                    asset_type=asset_type,
                                    N=N_int,
                                    q=q,
                                    r_f=r_f,
                                )
                            except Exception:
                                return np.nan

                    try:
                        iv_market = implied_volatility(
                            market_price=market_premium, price_of_sigma=price_of_sigma
                        )
                        if model == "Binomial Tree":
                            N_int = cast(int, N)
                            if (N is None) or (T is None) or (T <= 0):
                                sigma_floor = 1e-4
                                N_int = 200  # a safe fallback for IV solver callback
                            else:
                                N_int = int(N)
                                dt = T / N_int
                                # drift b depends on asset type
                                if asset_type == "currency":
                                    b = r - (r_f or 0.0)
                                elif asset_type == "future":
                                    b = 0.0
                                else:
                                    b = r - (q or 0.0)
                                sigma_floor = max(
                                    1e-4, 1.05 * abs(b) * np.sqrt(dt)
                                )  # 5% buffer
                            iv_market = implied_volatility(
                                market_price=market_premium,
                                price_of_sigma=price_of_sigma,
                                lo=sigma_floor,
                                lo_floor=sigma_floor,
                            )
                    except Exception as e:
                        st.warning(f"{e}")

                    if iv_market is not None:
                        st.caption(
                            f"Market IV ~ {iv_market*100:.2f}% from mid {market_premium:.4f}"
                        )
                        use_market_iv = st.checkbox(
                            "Use Market Implied Volatility", value=True
                        )
                        if use_market_iv:
                            sigma_eff = float(iv_market)
                    else:
                        st.warning("Could not solve implied volatility")
        else:
            st.warning("No expirations found; using manual inputs.")
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")

# part 2: compute premium
if T <= 0:
    st.error("Time to maturity must be positive")
    st.stop()

if sigma_eff < 0:  # Use sigma_eff since it's the effective volatility being used
    st.error("Volatility cannot be negative")
    st.stop()

if S_t <= 0:
    st.error("Stock price must be positive")
    st.stop()

if K <= 0:
    st.error("Strike price must be positive")
    st.stop()
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
                if not use_live:
                    st.success(f"Option Price: **{result['Price']:.4f}**")
                    st.subheader("Option Greeks")
                    st.table({k: [v] for k, v in result.items() if k != "Price"})
                    display_pnl(
                        option_price=result["Price"],
                        K=K,
                        option_type=option_type,
                        position_type=position_type,
                        number_of_contracts=number_of_contracts,
                        contract_multiplier=contract_multiplier,
                    )
                elif use_live and ticker:
                    market_premium = None
                    try:
                        if df_chain is not None:
                            market_premium = _market_premium_for_strike(
                                df=df_chain, K=K
                            )
                    except Exception as e:
                        st.warning(f"Could not get market premium for K={K}: {e}")

                    # recompute model result with the effective sigma
                    result = black_scholes_with_greeks(
                        S_t=S_t,
                        K=K,
                        T=T,
                        r=r,
                        sigma=sigma_eff,
                        q=q or 0.0,
                        option_type=option_type,
                    )

                    st.success(f"Option Price (Model): **{result['Price']:.4f}**")

                    if market_premium is not None:
                        st.info(f"Option Price (Market): **{market_premium:.4f}**")

                        # greeks
                        option_greeks = {
                            k: float(v) for k, v, in result.items() if k != "Price"
                        }
                        position_greeks = compute_position_greeks(
                            option_greeks=option_greeks,
                            position_type=position_type,
                            number_of_contracts=number_of_contracts,
                            contract_multiplier=contract_multiplier,
                        )
                        st.subheader("Option Greeks (Model)")
                        st.table({k: [v] for k, v in option_greeks.items()})
                        st.subheader("Position Greeks (Model)")
                        st.table({k: [v] for k, v in position_greeks.items()})

                        # Overlay P&L: model vs market
                        display_pnl_overlay(
                            option_price_model=result["Price"],
                            option_price_market=market_premium,
                            K=K,
                            option_type=option_type,
                            position_type=position_type,
                            number_of_contracts=number_of_contracts,
                            contract_multiplier=contract_multiplier,
                        )
                    else:
                        st.warning(
                            "No market premium found for this strike (off-grid or missing quotes)."
                        )
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
            N_int = cast(int, N)
            if sigma == 0.0:
                st.error("σ > 0 for the Binomial Tree model.")
                st.stop()
            price = dp_binomial_tree(
                S_t=S_t,
                K=K,
                T=T,
                r=r,
                sigma=sigma,
                option_type=option_type,
                exercise_type=exercise_type,
                asset_type=asset_type,
                N=N_int,
                q=q,
                r_f=r_f,
            )
            if not use_live:
                st.success(f"Option Price: **{price:.4f}**")
                display_pnl(
                    option_price=price,
                    K=K,
                    option_type=option_type,
                    position_type=position_type,
                    number_of_contracts=number_of_contracts,
                    contract_multiplier=contract_multiplier,
                )
            elif use_live and ticker:
                market_premium = None
                try:
                    if df_chain is not None:
                        market_premium = _market_premium_for_strike(df=df_chain, K=K)
                except Exception as e:
                    st.warning(f"Could not get market premium for K={K}: {e}")

                # recompute model result with the effective sigma
                price = dp_binomial_tree(
                    S_t=S_t,
                    K=K,
                    T=T,
                    r=r,
                    sigma=sigma_eff,
                    option_type=option_type,
                    exercise_type=exercise_type,
                    asset_type=asset_type,
                    N=N_int,
                    q=q,
                    r_f=r_f,
                )
                st.success(f"Option Price (Model): **{price:.4f}**")

                if market_premium is not None:
                    st.info(f"Option Price (Market): **{market_premium:.4f}**")
                    # overlay P&L: model vs market
                    display_pnl_overlay(
                        option_price_model=price,
                        option_price_market=market_premium,
                        K=K,
                        option_type=option_type,
                        position_type=position_type,
                        number_of_contracts=number_of_contracts,
                        contract_multiplier=contract_multiplier,
                    )
            else:
                st.warning(
                    "No market premium found for this strike (off-grid or missing quotes)."
                )
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
