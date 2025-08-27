from datetime import datetime
from typing import Callable, Optional, cast

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from scipy.optimize import brentq

SECONDS_PER_YEAR = 365.25 * 24 * 3600.0


@st.cache_data(ttl=300, show_spinner=False)
def _last_close(ticker: str) -> float:
    t = yf.Ticker(ticker)
    hist = t.history(period="1d")
    if hist.empty:
        raise ValueError("No price data for ticker.")
    return float(hist["Close"].iloc[-1])


@st.cache_data(ttl=300, show_spinner=False)
def _expirations(ticker: str):
    return yf.Ticker(ticker).options


@st.cache_resource(ttl=300, show_spinner=False)
def _option_chain(ticker: str, expiry: str):
    return yf.Ticker(ticker).option_chain(expiry)


def _yearfrac(expiry: datetime) -> float:
    return max((expiry - datetime.now()).total_seconds() / SECONDS_PER_YEAR, 1e-8)


def _market_premium_for_strike(df: pd.DataFrame, K: float, *, tol: float | None = None):
    if df is None or df.empty or "strike" not in df.columns:
        return None

    # clean strike -> numeric and filter valid rows
    strikes_num = pd.to_numeric(df["strike"], errors="coerce")
    valid_mask = strikes_num.notna()
    if not valid_mask.any():
        return None

    dff = df.loc[valid_mask].copy()
    strikes = strikes_num.loc[valid_mask].to_numpy(float)

    # choose a sensible tolerance if none given: half the median strike step, capped at $0.05
    if tol is None:
        tol = 0.01  # default 1¢
        uniq = np.unique(np.round(strikes, 6))
        if uniq.size > 1:
            diffs = np.diff(uniq.astype(float))
            diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
            if diffs.size:
                step = float(np.median(diffs).item())
                half_step = float(0.5 * step)
                tol = float(max(1e-6, min(0.05, half_step)))

    # nearest strike, but only accept if it's "close enough"
    i = int(np.argmin(np.abs(strikes - float(K))))
    if np.abs(strikes[i] - float(K)) > tol:
        return None  # treat as no matching strike in this expiry

    row = dff.iloc[i]

    # safely fetch quote fields
    def fget(series, col):
        if col in series.index:
            try:
                v = float(series[col])
                return v if np.isfinite(v) else np.nan
            except Exception:
                return np.nan
        return np.nan

    bid = fget(row, "bid")
    ask = fget(row, "ask")
    last = fget(
        row, "lastPrice"
    )  # yfinance uses 'lastPrice'; add aliases if your feed differs

    # prefer mid if both sides are sane; else fall back to last
    if np.isfinite(bid) and np.isfinite(ask) and ask > 0 and 0 <= bid <= ask:
        return 0.5 * (bid + ask)
    if np.isfinite(last) and last > 0:
        return last

    return None


def implied_volatility(
    market_price: float,
    price_of_sigma: Callable[[float], float],
    *,
    lo: float = 1e-4,
    hi: float = 5.0,
    xtol: float = 1e-10,
    max_iter: int = 200,
    expand_factor: float = 1.5,
    shrink_factor: float = 0.5,
    max_expands: int = 12,
    lo_floor: float = 1e-8,
    hi_cap: float = 10.0
) -> Optional[float]:
    if market_price is None or not np.isfinite(market_price) or market_price <= 0:
        return None

    def f(sig: float) -> float:
        return price_of_sigma(sig) - market_price

    f_lo = f(lo)
    f_hi = f(hi)

    # expand upper bound if price at hi is above market (f_hi < 0)
    expands = 0
    while (
        np.isfinite(f_hi) and (f_hi < 0.0) and (hi < hi_cap) and (expands < max_expands)
    ):
        hi *= expand_factor
        f_hi = f(hi)
        expands += 1

    # contract lower bound if price at lo is already above market (f_lo > 0)
    shrinks = 0
    while (
        np.isfinite(f_lo)
        and (f_lo > 0.0)
        and (lo > lo_floor)
        and (shrinks < max_expands)
    ):
        lo *= shrink_factor
        f_lo = f(lo)
        shrinks += 1

    # require sign change for brentq
    if not (
        np.isfinite(f_lo) and np.isfinite(f_hi) and (np.sign(f_lo) != np.sign(f_hi))
    ):
        return None

    try:
        return cast(float, (brentq(f, lo, hi, xtol=xtol, maxiter=max_iter)))
    except Exception:
        return None


def compute_position_greeks(
    option_greeks: dict[str, float],
    position_type,
    number_of_contracts,
    contract_multiplier,
):
    sign = 1.0 if position_type == "long" else -1.0
    scale = sign * number_of_contracts * contract_multiplier
    return {greek: val * scale for greek, val, in option_greeks.items()}
