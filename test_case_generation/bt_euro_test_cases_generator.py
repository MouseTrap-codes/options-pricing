import argparse
import json
from typing import Dict

import QuantLib as ql

from .bt_euro_test_inputs import (
    EUROPEAN_CALL_INPUTS,
    EUROPEAN_EDGE_CASES,
    EUROPEAN_PUT_INPUTS,
)


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
    day_count = ql.Actual365Fixed()
    cal = ql.NullCalendar()
    today = ql.Date.todaysDate()
    ql.Settings.instance().evaluationDate = today

    # dividend/foreign-rate logic
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
        # crr tree
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


def build_payload_binomial_euro() -> Dict[str, list]:
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
        "--bt-out",
        default="tests/data/bt_european_test_cases.json",
        help="Path for binomial European test cases JSON",
    )
    args = parser.parse_args()

    # Binomial-European cases
    bt_payload = build_payload_binomial_euro()
    with open(args.bt_out, "w", encoding="utf-8") as f:
        json.dump(bt_payload, f, indent=2)
    print(f"Wrote {args.bt_out}")
