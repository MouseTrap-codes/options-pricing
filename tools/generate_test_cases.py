import argparse
import json
import math
from typing import Dict, List, Tuple

import QuantLib as ql


def european_exact(
    S: float, K: float, T: float, r: float, q: float, sigma: float, is_call: bool
) -> Dict[str, float]:
    # Map BSM(q) -> Black(76) inputs
    F = S * math.exp((r - q) * T)
    df = math.exp(-r * T)
    stddev = sigma * math.sqrt(T)

    payoff = ql.PlainVanillaPayoff(ql.Option.Call if is_call else ql.Option.Put, K)
    calc = ql.BlackCalculator(payoff, F, stddev, df)  # available in QuantLib-Python

    price = float(calc.value())

    # Closed-form Greeks in BSM with continuous yield q (to match your model’s units)
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


def make_case(name: str, inputs: dict) -> dict:
    ql_args = adapt_inputs_to_quantlib(inputs)
    exp = round6(european_exact(**ql_args))
    return {"name": name, "inputs": inputs, "expected": exp}


def build_payload():
    calls = [make_case(n, x) for n, x in CALL_INPUTS]
    puts = [make_case(n, x) for n, x in PUT_INPUTS]
    return {"CALL_CASES": calls, "PUT_CASES": puts}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="tests/data/bs_cases.json")
    args = parser.parse_args()

    payload = build_payload()
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {args.out}")
