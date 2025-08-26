from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

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
