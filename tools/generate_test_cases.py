import argparse
import json
import math
from typing import Dict, List, Tuple

import QuantLib as ql


def european_exact(
    S: float, K: float, T: float, r: float, q: float, sigma: float, is_call: bool
) -> Dict[str, float]:
    # map BSM(q) -> Black(76) inputs
    F = S * math.exp((r - q) * T)
    df = math.exp(-r * T)
    stddev = sigma * math.sqrt(T)

    payoff = ql.PlainVanillaPayoff(ql.Option.Call if is_call else ql.Option.Put, K)
    calc = ql.BlackCalculator(payoff, F, stddev, df)  # available in QuantLib-Python

    price = float(calc.value())

    # closed-form Greeks in BSM with continuous yield q (to match your model’s units)
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return {
            "Price": price,
            "Delta": 0.0,
            "Gamma": 0.0,
            "Vega": 0.0,
            "Theta": 0.0,
            "Rho": 0.0,
        }

    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    # standard normal pdf / cdf
    nd1 = math.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi)

    def cdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    if is_call:
        delta = math.exp(-q * T) * cdf(d1)
        theta = (
            -(S * math.exp(-q * T) * nd1 * sigma) / (2.0 * math.sqrt(T))
            - r * K * df * cdf(d2)
            + q * S * math.exp(-q * T) * cdf(d1)
        ) / 365.0
        rho = (K * T * df * cdf(d2)) / 100.0
    else:
        delta = -math.exp(-q * T) * cdf(-d1)
        theta = (
            -(S * math.exp(-q * T) * nd1 * sigma) / (2.0 * math.sqrt(T))
            + r * K * df * cdf(-d2)
            - q * S * math.exp(-q * T) * cdf(-d1)
        ) / 365.0
        rho = (-K * T * df * cdf(-d2)) / 100.0

    gamma = (math.exp(-q * T) * nd1) / (S * sigma * math.sqrt(T))
    vega = (S * math.exp(-q * T) * nd1 * math.sqrt(T)) / 100.0  # per 1%

    return {
        "Price": price,
        "Delta": delta,
        "Gamma": gamma,
        "Vega": vega,
        "Theta": theta,
        "Rho": rho,
    }


CALL_INPUTS: List[Tuple[str, dict]] = [
    (
        "Hull_call_classic",
        {
            "S_t": 42.0,
            "K": 40.0,
            "T": 0.5,
            "r": 0.10,
            "sigma": 0.20,
            "q": 0.0,
            "option_type": "call",
        },
    ),
    (
        "ATM_call_baseline",
        {
            "S_t": 100.0,
            "K": 100.0,
            "T": 1.0,
            "r": 0.05,
            "sigma": 0.20,
            "q": 0.0,
            "option_type": "call",
        },
    ),
    (
        "Dividend_ATM_call",
        {
            "S_t": 50.0,
            "K": 50.0,
            "T": 0.5,
            "r": 0.06,
            "sigma": 0.30,
            "q": 0.03,
            "option_type": "call",
        },
    ),
    (
        "Deep_ITM_call_long_T",
        {
            "S_t": 150.0,
            "K": 100.0,
            "T": 2.0,
            "r": 0.02,
            "sigma": 0.25,
            "q": 0.0,
            "option_type": "call",
        },
    ),
    (
        "Short_maturity_call_OTM",
        {
            "S_t": 100.0,
            "K": 105.0,
            "T": 1.0 / 12.0,
            "r": 0.01,
            "sigma": 0.15,
            "q": 0.0,
            "option_type": "call",
        },
    ),
    (
        "High_vol_call_with_q",
        {
            "S_t": 100.0,
            "K": 100.0,
            "T": 1.0,
            "r": 0.03,
            "sigma": 0.35,
            "q": 0.05,
            "option_type": "call",
        },
    ),
    (
        "Low_rate_call",
        {
            "S_t": 120.0,
            "K": 110.0,
            "T": 0.75,
            "r": 0.005,
            "sigma": 0.18,
            "q": 0.0,
            "option_type": "call",
        },
    ),
    (
        "Neg_rate_call",
        {
            "S_t": 100.0,
            "K": 90.0,
            "T": 1.5,
            "r": -0.01,
            "sigma": 0.22,
            "q": 0.0,
            "option_type": "call",
        },
    ),
    (
        "Near_expiry_call_ATM",
        {
            "S_t": 100.0,
            "K": 100.0,
            "T": 1.0 / 365.0,
            "r": 0.01,
            "sigma": 0.30,
            "q": 0.0,
            "option_type": "call",
        },
    ),
    (
        "Near_expiry_call_ITM",
        {
            "S_t": 105.0,
            "K": 100.0,
            "T": 1.0 / 365.0,
            "r": 0.01,
            "sigma": 0.30,
            "q": 0.0,
            "option_type": "call",
        },
    ),
    (
        "Near_expiry_call_OTM",
        {
            "S_t": 95.0,
            "K": 100.0,
            "T": 1.0 / 365.0,
            "r": 0.01,
            "sigma": 0.30,
            "q": 0.0,
            "option_type": "call",
        },
    ),
    (
        "Very_high_vol_call",
        {
            "S_t": 100.0,
            "K": 100.0,
            "T": 1.0,
            "r": 0.02,
            "sigma": 1.0,
            "q": 0.0,
            "option_type": "call",
        },
    ),
    (
        "Deep_OTM_long_call",
        {
            "S_t": 50.0,
            "K": 150.0,
            "T": 3.0,
            "r": 0.02,
            "sigma": 0.30,
            "q": 0.0,
            "option_type": "call",
        },
    ),
    (
        "High_dividend_call",
        {
            "S_t": 100.0,
            "K": 100.0,
            "T": 0.75,
            "r": 0.03,
            "sigma": 0.25,
            "q": 0.10,
            "option_type": "call",
        },
    ),
    (
        "Zero_rates_call",
        {
            "S_t": 100.0,
            "K": 90.0,
            "T": 0.5,
            "r": 0.0,
            "sigma": 0.20,
            "q": 0.0,
            "option_type": "call",
        },
    ),
    (
        "Low_vol_ATM_call",
        {
            "S_t": 100.0,
            "K": 100.0,
            "T": 1.0,
            "r": 0.03,
            "sigma": 0.05,
            "q": 0.0,
            "option_type": "call",
        },
    ),
]

PUT_INPUTS: List[Tuple[str, dict]] = [
    (
        "ATM_put_baseline",
        {
            "S_t": 100.0,
            "K": 100.0,
            "T": 1.0,
            "r": 0.05,
            "sigma": 0.20,
            "q": 0.0,
            "option_type": "put",
        },
    ),
    (
        "McDonald_like_put",
        {
            "S_t": 40.0,
            "K": 45.0,
            "T": 0.25,
            "r": 0.05,
            "sigma": 0.25,
            "q": 0.0,
            "option_type": "put",
        },
    ),
    (
        "Dividend_ATM_put",
        {
            "S_t": 50.0,
            "K": 50.0,
            "T": 0.5,
            "r": 0.06,
            "sigma": 0.30,
            "q": 0.03,
            "option_type": "put",
        },
    ),
    (
        "Deep_ITM_put_long_T",
        {
            "S_t": 80.0,
            "K": 100.0,
            "T": 2.0,
            "r": 0.01,
            "sigma": 0.25,
            "q": 0.0,
            "option_type": "put",
        },
    ),
    (
        "Short_maturity_put_ITM",
        {
            "S_t": 100.0,
            "K": 105.0,
            "T": 1.0 / 12.0,
            "r": 0.01,
            "sigma": 0.15,
            "q": 0.0,
            "option_type": "put",
        },
    ),
    (
        "OTM_put",
        {
            "S_t": 120.0,
            "K": 100.0,
            "T": 0.5,
            "r": 0.02,
            "sigma": 0.20,
            "q": 0.0,
            "option_type": "put",
        },
    ),
    (
        "High_dividend_put",
        {
            "S_t": 100.0,
            "K": 100.0,
            "T": 1.0,
            "r": 0.03,
            "sigma": 0.25,
            "q": 0.06,
            "option_type": "put",
        },
    ),
    (
        "Neg_rate_put",
        {
            "S_t": 100.0,
            "K": 110.0,
            "T": 1.5,
            "r": -0.01,
            "sigma": 0.22,
            "q": 0.0,
            "option_type": "put",
        },
    ),
    (
        "Near_expiry_put_ATM",
        {
            "S_t": 100.0,
            "K": 100.0,
            "T": 1.0 / 365.0,
            "r": 0.01,
            "sigma": 0.30,
            "q": 0.0,
            "option_type": "put",
        },
    ),
    (
        "Near_expiry_put_ITM",
        {
            "S_t": 95.0,
            "K": 100.0,
            "T": 1.0 / 365.0,
            "r": 0.01,
            "sigma": 0.30,
            "q": 0.0,
            "option_type": "put",
        },
    ),
    (
        "Near_expiry_put_OTM",
        {
            "S_t": 105.0,
            "K": 100.0,
            "T": 1.0 / 365.0,
            "r": 0.01,
            "sigma": 0.30,
            "q": 0.0,
            "option_type": "put",
        },
    ),
    (
        "Very_high_vol_put",
        {
            "S_t": 100.0,
            "K": 100.0,
            "T": 1.0,
            "r": 0.02,
            "sigma": 1.0,
            "q": 0.0,
            "option_type": "put",
        },
    ),
    (
        "Deep_OTM_put_long",
        {
            "S_t": 150.0,
            "K": 50.0,
            "T": 3.0,
            "r": 0.02,
            "sigma": 0.30,
            "q": 0.0,
            "option_type": "put",
        },
    ),
    (
        "High_dividend_put_shortT",
        {
            "S_t": 100.0,
            "K": 90.0,
            "T": 0.75,
            "r": 0.03,
            "sigma": 0.25,
            "q": 0.08,
            "option_type": "put",
        },
    ),
    (
        "Zero_rates_put",
        {
            "S_t": 90.0,
            "K": 100.0,
            "T": 0.5,
            "r": 0.0,
            "sigma": 0.20,
            "q": 0.0,
            "option_type": "put",
        },
    ),
    (
        "Low_vol_ATM_put",
        {
            "S_t": 100.0,
            "K": 100.0,
            "T": 1.0,
            "r": 0.03,
            "sigma": 0.05,
            "q": 0.0,
            "option_type": "put",
        },
    ),
]

EUROPEAN_CALL_INPUTS: List[Tuple[str, dict]] = [
    (
        "Hull_classic_european_call",
        {
            "sigma": 0.20,
            "T": 0.25,
            "N": 100,
            "S_t": 42.0,
            "r": 0.10,
            "K": 40.0,
            "q": None,
            "r_f": None,
            "asset_type": "nondividend",
            "exercise_type": "european",
            "option_type": "call",
        },
    ),
    (
        "ATM_european_call",
        {
            "sigma": 0.20,
            "T": 1.0,
            "N": 200,
            "S_t": 100.0,
            "r": 0.05,
            "K": 100.0,
            "q": None,
            "r_f": None,
            "asset_type": "nondividend",
            "exercise_type": "european",
            "option_type": "call",
        },
    ),
    (
        "Dividend_european_call",
        {
            "sigma": 0.30,
            "T": 0.5,
            "N": 150,
            "S_t": 50.0,
            "r": 0.06,
            "K": 50.0,
            "q": 0.03,
            "r_f": None,
            "asset_type": "dividend",
            "exercise_type": "european",
            "option_type": "call",
        },
    ),
    (
        "Currency_european_call",
        {
            "sigma": 0.15,
            "T": 1.0,
            "N": 200,
            "S_t": 1.25,  # EUR/USD
            "r": 0.02,  # USD rate
            "K": 1.20,
            "q": None,
            "r_f": 0.01,  # EUR rate
            "asset_type": "currency",
            "exercise_type": "european",
            "option_type": "call",
        },
    ),
    (
        "Future_european_call",
        {
            "sigma": 0.25,
            "T": 0.75,
            "N": 100,
            "S_t": 2500.0,  # Future price
            "r": 0.03,
            "K": 2400.0,
            "q": None,
            "r_f": None,
            "asset_type": "future",
            "exercise_type": "european",
            "option_type": "call",
        },
    ),
    (
        "High_vol_european_call",
        {
            "sigma": 0.80,
            "T": 1.0,
            "N": 250,
            "S_t": 100.0,
            "r": 0.05,
            "K": 100.0,
            "q": None,
            "r_f": None,
            "asset_type": "nondividend",
            "exercise_type": "european",
            "option_type": "call",
        },
    ),
    (
        "Short_term_european_call",
        {
            "sigma": 0.20,
            "T": 1.0 / 52.0,  # 1 week
            "N": 50,
            "S_t": 100.0,
            "r": 0.01,
            "K": 102.0,
            "q": None,
            "r_f": None,
            "asset_type": "nondividend",
            "exercise_type": "european",
            "option_type": "call",
        },
    ),
    (
        "Deep_ITM_european_call",
        {
            "sigma": 0.25,
            "T": 2.0,
            "N": 300,
            "S_t": 150.0,
            "r": 0.04,
            "K": 100.0,
            "q": None,
            "r_f": None,
            "asset_type": "nondividend",
            "exercise_type": "european",
            "option_type": "call",
        },
    ),
    (
        "Low_vol_dividend_european_call",
        {
            "sigma": 0.10,
            "T": 1.0,
            "N": 200,
            "S_t": 100.0,
            "r": 0.03,
            "K": 95.0,
            "q": 0.02,
            "r_f": None,
            "asset_type": "dividend",
            "exercise_type": "european",
            "option_type": "call",
        },
    ),
    (
        "Negative_rate_european_call",
        {
            "sigma": 0.20,
            "T": 1.0,
            "N": 150,
            "S_t": 100.0,
            "r": -0.005,
            "K": 95.0,
            "q": None,
            "r_f": None,
            "asset_type": "nondividend",
            "exercise_type": "european",
            "option_type": "call",
        },
    ),
]

EUROPEAN_PUT_INPUTS: List[Tuple[str, dict]] = [
    (
        "McDonald_european_put",
        {
            "sigma": 0.25,
            "T": 0.25,
            "N": 100,
            "S_t": 40.0,
            "r": 0.05,
            "K": 45.0,
            "q": None,
            "r_f": None,
            "asset_type": "nondividend",
            "exercise_type": "european",
            "option_type": "put",
        },
    ),
    (
        "ATM_european_put",
        {
            "sigma": 0.20,
            "T": 1.0,
            "N": 200,
            "S_t": 100.0,
            "r": 0.05,
            "K": 100.0,
            "q": None,
            "r_f": None,
            "asset_type": "nondividend",
            "exercise_type": "european",
            "option_type": "put",
        },
    ),
    (
        "Dividend_european_put",
        {
            "sigma": 0.30,
            "T": 0.5,
            "N": 150,
            "S_t": 50.0,
            "r": 0.06,
            "K": 50.0,
            "q": 0.03,
            "r_f": None,
            "asset_type": "dividend",
            "exercise_type": "european",
            "option_type": "put",
        },
    ),
    (
        "Currency_european_put",
        {
            "sigma": 0.12,
            "T": 0.5,
            "N": 100,
            "S_t": 1.15,  # EUR/USD
            "r": 0.025,  # USD rate
            "K": 1.20,
            "q": None,
            "r_f": 0.015,  # EUR rate
            "asset_type": "currency",
            "exercise_type": "european",
            "option_type": "put",
        },
    ),
    (
        "Future_european_put",
        {
            "sigma": 0.35,
            "T": 0.5,
            "N": 100,
            "S_t": 2200.0,
            "r": 0.02,
            "K": 2300.0,
            "q": None,
            "r_f": None,
            "asset_type": "future",
            "exercise_type": "european",
            "option_type": "put",
        },
    ),
    (
        "Deep_ITM_european_put",
        {
            "sigma": 0.30,
            "T": 1.5,
            "N": 250,
            "S_t": 80.0,
            "r": 0.03,
            "K": 120.0,
            "q": None,
            "r_f": None,
            "asset_type": "nondividend",
            "exercise_type": "european",
            "option_type": "put",
        },
    ),
    (
        "High_dividend_european_put",
        {
            "sigma": 0.25,
            "T": 1.0,
            "N": 200,
            "S_t": 100.0,
            "r": 0.04,
            "K": 100.0,
            "q": 0.08,
            "r_f": None,
            "asset_type": "dividend",
            "exercise_type": "european",
            "option_type": "put",
        },
    ),
    (
        "Short_term_european_put",
        {
            "sigma": 0.40,
            "T": 1.0 / 12.0,  # 1 month
            "N": 30,
            "S_t": 100.0,
            "r": 0.02,
            "K": 100.0,
            "q": None,
            "r_f": None,
            "asset_type": "nondividend",
            "exercise_type": "european",
            "option_type": "put",
        },
    ),
    (
        "Very_low_vol_european_put",
        {
            "sigma": 0.05,
            "T": 1.0,
            "N": 150,
            "S_t": 95.0,
            "r": 0.03,
            "K": 100.0,
            "q": None,
            "r_f": None,
            "asset_type": "nondividend",
            "exercise_type": "european",
            "option_type": "put",
        },
    ),
    (
        "Long_term_OTM_european_put",
        {
            "sigma": 0.20,
            "T": 3.0,
            "N": 500,
            "S_t": 120.0,
            "r": 0.04,
            "K": 100.0,
            "q": None,
            "r_f": None,
            "asset_type": "nondividend",
            "exercise_type": "european",
            "option_type": "put",
        },
    ),
]

# Edge cases for robustness testing - European only
EUROPEAN_EDGE_CASES: List[Tuple[str, dict]] = [
    (
        "Min_time_steps_european",
        {
            "sigma": 0.20,
            "T": 0.25,
            "N": 1,  # Minimum steps
            "S_t": 100.0,
            "r": 0.05,
            "K": 100.0,
            "q": None,
            "r_f": None,
            "asset_type": "nondividend",
            "exercise_type": "european",
            "option_type": "call",
        },
    ),
    (
        "Very_short_expiry_european",
        {
            "sigma": 0.20,
            "T": 1.0 / 365.0,  # 1 day
            "N": 10,
            "S_t": 100.0,
            "r": 0.05,
            "K": 100.0,
            "q": None,
            "r_f": None,
            "asset_type": "nondividend",
            "exercise_type": "european",
            "option_type": "put",
        },
    ),
    (
        "High_time_steps_european",
        {
            "sigma": 0.25,
            "T": 1.0,
            "N": 1000,  # Many steps for convergence testing
            "S_t": 100.0,
            "r": 0.03,
            "K": 105.0,
            "q": None,
            "r_f": None,
            "asset_type": "nondividend",
            "exercise_type": "european",
            "option_type": "call",
        },
    ),
    (
        "Zero_interest_rate_european",
        {
            "sigma": 0.30,
            "T": 1.0,
            "N": 100,
            "S_t": 100.0,
            "r": 0.0,
            "K": 100.0,
            "q": None,
            "r_f": None,
            "asset_type": "nondividend",
            "exercise_type": "european",
            "option_type": "call",
        },
    ),
    (
        "Very_high_volatility_european",
        {
            "sigma": 2.0,  # 200% vol
            "T": 1.0,
            "N": 200,
            "S_t": 100.0,
            "r": 0.05,
            "K": 100.0,
            "q": None,
            "r_f": None,
            "asset_type": "nondividend",
            "exercise_type": "european",
            "option_type": "call",
        },
    ),
]


def adapt_inputs_to_quantlib(inputs: dict) -> dict:
    return {
        "S": inputs["S_t"],  # map S_t -> S
        "K": inputs["K"],
        "T": inputs["T"],
        "r": inputs["r"],
        "q": inputs["q"],
        "sigma": inputs["sigma"],
        "is_call": inputs["option_type"].lower()
        == "call",  # option_type -> is_call (bool)
    }


def round6(d: Dict[str, float]) -> Dict[str, float]:
    return {k: round(float(v), 6) for k, v in d.items()}


def make_bsm_case(name: str, inputs: dict) -> dict:
    ql_args = adapt_inputs_to_quantlib(inputs)
    exp = round6(european_exact(**ql_args))
    return {"name": name, "inputs": inputs, "expected": exp}


def build_payload_bsm():
    calls = [make_bsm_case(n, x) for n, x in CALL_INPUTS]
    puts = [make_bsm_case(n, x) for n, x in PUT_INPUTS]
    return {"CALL_CASES": calls, "PUT_CASES": puts}


def binomial_european_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    *,
    q: float | None = None,
    asset_type: str = "nondividend",
    r_f: float | None = None,
    N: int = 100,
    is_call: bool = True,
) -> float:
    # Setup dates / market data
    day_count = ql.Actual365Fixed()
    cal = ql.NullCalendar()
    today = ql.Date.todaysDate()
    ql.Settings.instance().evaluationDate = today

    # dividend/foreign-rate logic to match your dp pricer semantics
    if asset_type == "currency":
        if r_f is None:
            raise ValueError("asset_type='currency' requires r_f")
        div_rate = r_f
    elif asset_type == "future":
        # driftless under risk-neutral measure
        div_rate = r
    else:
        div_rate = 0.0 if q is None else float(q)

    spot = ql.QuoteHandle(ql.SimpleQuote(S))
    divTS = ql.YieldTermStructureHandle(ql.FlatForward(today, div_rate, day_count))
    rfTS = ql.YieldTermStructureHandle(ql.FlatForward(today, r, day_count))
    volTS = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(today, cal, sigma, day_count)
    )

    process = ql.BlackScholesMertonProcess(spot, divTS, rfTS, volTS)

    typ = ql.Option.Call if is_call else ql.Option.Put
    payoff = ql.PlainVanillaPayoff(typ, K)
    # ensure at least 1 day into the future
    maturity_days = max(1, int(round(T * 365)))
    exercise = ql.EuropeanExercise(today + maturity_days)

    opt = ql.VanillaOption(payoff, exercise)

    steps = max(2, int(N))
    try:
        # Cox–Ross–Rubinstein tree
        opt.setPricingEngine(
            ql.BinomialVanillaEngine(process, "CoxRossRubinstein", steps=steps)
        )
    except TypeError:
        opt.setPricingEngine(ql.BinomialVanillaEngine(process, "crr", steps=steps))

    return float(opt.NPV())


def make_bt_case(name: str, inputs: dict) -> dict:
    price = binomial_european_price(
        S=inputs["S_t"],
        K=inputs["K"],
        T=inputs["T"],
        r=inputs["r"],
        sigma=inputs["sigma"],
        q=inputs.get("q"),
        asset_type=inputs.get("asset_type", "nondividend"),
        r_f=inputs.get("r_f"),
        N=int(inputs["N"]),
        is_call=(inputs["option_type"].lower() == "call"),
    )
    return {
        "name": name,
        "inputs": inputs,
        "expected": {"Price": round(float(price), 6)},
    }


def build_payload_binomial() -> Dict[str, list]:
    calls = [make_bt_case(n, x) for n, x in EUROPEAN_CALL_INPUTS]
    puts = [make_bt_case(n, x) for n, x in EUROPEAN_PUT_INPUTS]
    edges = [make_bt_case(n, x) for n, x in EUROPEAN_EDGE_CASES]
    return {
        "EUROPEAN_CALLS": calls,
        "EUROPEAN_PUTS": puts,
        "EUROPEAN_EDGE_CASES": edges,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="tests/data/bs_cases.json",
        help="Path for Black–Scholes test cases JSON",
    )
    parser.add_argument(
        "--bt-out",
        default="tests/data/bt_european_test_cases.json",
        help="Path for binomial European test cases JSON",
    )
    args = parser.parse_args()

    # BSM cases
    bsm_payload = build_payload_bsm()
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(bsm_payload, f, indent=2)
    print(f"Wrote {args.out}")

    # Binomial-European cases
    bt_payload = build_payload_binomial()
    with open(args.bt_out, "w", encoding="utf-8") as f:
        json.dump(bt_payload, f, indent=2)
    print(f"Wrote {args.bt_out}")
