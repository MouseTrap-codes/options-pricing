import json
from pathlib import Path

import numpy as np
import pytest

from models import dp_binomial_tree
from tests.tolerances import BT_AMER_TOL_OVERRIDES, BT_EURO_TOL_OVERRIDES, BT_TOL_EURO


def _load_bt_euro_test_cases():
    path = Path(__file__).parent / "data" / "bt_european_test_cases.json"
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    test_cases = []
    for section in ("EUROPEAN_CALLS", "EUROPEAN_PUTS", "EUROPEAN_EDGE_CASES"):
        for test_case in payload.get(section, []):
            test_cases.append(test_case)
    return test_cases


def _load_bt_amer_test_cases():
    path = Path(__file__).parent / "data" / "bt_american_test_cases.json"
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    test_cases = []
    for section in ("AMERICAN_CALLS", "AMERICAN_PUTS", "AMERICAN_EDGE_CASES"):
        for test_case in payload.get(section, []):
            test_cases.append(test_case)
    return test_cases


def _normalize_inputs(d: dict) -> dict:
    """Match QuantLib’s engine conventions used to generate the JSON."""
    x = dict(d)

    days = max(1, int(round(float(x["T"]) * 365.0)))
    x["T"] = days / 365.0

    if int(x.get("N", 2)) < 2:
        x["N"] = 2

    if x.get("asset_type") != "currency" and x.get("q", None) is None:
        x["q"] = 0.0

    return x


@pytest.mark.parametrize(
    "test_case", _load_bt_euro_test_cases(), ids=lambda c: c["name"]
)
def test_binomial_european_price(test_case):
    inputs = _normalize_inputs(test_case["inputs"])
    price = dp_binomial_tree(**inputs)
    exp = test_case["expected"]["Price"]
    tol = (
        BT_AMER_TOL_OVERRIDES.get(test_case["name"])
        or BT_EURO_TOL_OVERRIDES.get(test_case["name"])
        or BT_TOL_EURO
    )
    assert price == pytest.approx(exp, rel=tol["rel"], abs=tol["abs"])


@pytest.mark.parametrize(
    "test_case", _load_bt_amer_test_cases(), ids=lambda c: c["name"]
)
def test_binomial_american_price(test_case):
    inputs = _normalize_inputs(test_case["inputs"])
    price = dp_binomial_tree(**inputs)
    exp = test_case["expected"]["Price"]
    tol = (
        BT_AMER_TOL_OVERRIDES.get(test_case["name"])
        or BT_EURO_TOL_OVERRIDES.get(test_case["name"])
        or BT_TOL_EURO
    )
    assert price == pytest.approx(exp, rel=tol["rel"], abs=tol["abs"])


def test_invalid_steps_raises():
    # simple ATM call with invalid N=0 should raise
    with pytest.raises(ValueError):
        dp_binomial_tree(
            sigma=0.2,
            T=1.0,
            N=0,  # invalid
            S_t=100.0,
            r=0.05,
            K=100.0,
            q=None,
            r_f=None,
            asset_type="nondividend",
            exercise_type="european",
            option_type="call",
        )


# inputs and outputs for comparisons
hull_params = {
    "inputs": {
        "S_t": 42.0,  # current stock price
        "K": 40.0,  # strike price
        "T": 0.25,  # 3 months = 0.25 years
        "r": 0.10,  # 10% risk-free rate
        "sigma": 0.20,  # 20% volatility
        "q": 0.0,  # no dividends
        "option_type": "call",
        "N": 100,
    },
    "expected": {
        "Price": 4.759,  # Hull's textbook value
    },
}


def test_invalid_inputs():
    # invalid option type
    with pytest.raises(ValueError):
        dp_binomial_tree(0, 0, 0, 0, 0, 0, option_type="invalid")

    # negative underlying price
    with pytest.raises(ValueError):
        dp_binomial_tree(-100, 0, 0, 0, 0, 0, option_type="call")

    # negative strike
    with pytest.raises(ValueError):
        dp_binomial_tree(0, -100, 0, 0, 0, 0, option_type="call")


# using example params from Hull's "Options, Futures, and Other Derivatives" -> highly recommend
def test_mathematical_properties():
    S_t = hull_params["inputs"]["S_t"]
    K = hull_params["inputs"]["K"]
    T = hull_params["inputs"]["T"]
    r = hull_params["inputs"]["r"]
    sigma = hull_params["inputs"]["sigma"]
    q = hull_params["inputs"]["q"]
    N = hull_params["inputs"]["N"]

    call = dp_binomial_tree(
        S_t=S_t, K=K, T=T, r=r, sigma=sigma, q=q, option_type="call", N=N
    )

    put = dp_binomial_tree(
        S_t=S_t, K=K, T=T, r=r, sigma=sigma, q=q, option_type="put", N=N
    )

    # put-call parity 0> C - P = S * e^(-q*T) - K*e^(-r*T)
    expected_parity = S_t * np.exp(-q * T) - K * np.exp(-r * T)
    actual_parity = call - put
    assert abs(actual_parity - expected_parity) < 1e-10
