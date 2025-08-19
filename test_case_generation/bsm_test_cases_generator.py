import argparse
import json
import math
from typing import Dict

import QuantLib as ql

from .bsm_test_inputs import CALL_INPUTS, PUT_INPUTS


def european_exact(
    S: float, K: float, T: float, r: float, q: float, sigma: float, is_call: bool
) -> Dict[str, float]:
    # map BSM(q) -> Black(76) inputs
    F = S * math.exp((r - q) * T)
    df = math.exp(-r * T)
    stddev = sigma * math.sqrt(T)

    payoff = ql.PlainVanillaPayoff(ql.Option.Call if is_call else ql.Option.Put, K)
    calc = ql.BlackCalculator(payoff, F, stddev, df)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="tests/data/bs_cases.json",
        help="Path for Black–Scholes test cases JSON",
    )

    args = parser.parse_args()

    bsm_payload = build_payload_bsm()
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(bsm_payload, f, indent=2)
    print(f"Wrote {args.out}")
