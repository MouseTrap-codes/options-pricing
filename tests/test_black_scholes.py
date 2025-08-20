import json
from pathlib import Path

import numpy as np
import pytest

from models import black_scholes_with_greeks
from tests.tolerances import BSM_TOL


def load_test_cases():
    path = Path(__file__).parent / "data" / "bs_cases.json"
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["CALL_CASES"] + payload["PUT_CASES"]


@pytest.mark.parametrize("test_case", load_test_cases(), ids=lambda c: c["name"])
def test_black_scholes_test_cases(test_case):
    got = black_scholes_with_greeks(**test_case["inputs"])
    for k, exp in test_case["expected"].items():
        tol = BSM_TOL[k]
        assert got[k] == pytest.approx(exp, rel=tol["rel"], abs=tol["abs"])


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
    },
    "expected": {
        "Price": 4.759,  # Hull's textbook value
    },
}


def test_invalid_inputs():
    # invalid option type
    with pytest.raises(ValueError):
        black_scholes_with_greeks(0, 0, 0, 0, 0, option_type="invalid")

    # negative underlying price
    with pytest.raises(ValueError):
        black_scholes_with_greeks(-100, 0, 0, 0, 0, option_type="call")

    # negative strike
    with pytest.raises(ValueError):
        black_scholes_with_greeks(0, -100, 0, 0, 0, option_type="call")


# using example params from Hull's "Options, Futures, and Other Derivatives" -> highly recommend
def test_mathematical_properties():
    S_t = hull_params["inputs"]["S_t"]
    K = hull_params["inputs"]["K"]
    T = hull_params["inputs"]["T"]
    r = hull_params["inputs"]["r"]
    sigma = hull_params["inputs"]["sigma"]
    q = hull_params["inputs"]["q"]

    call = black_scholes_with_greeks(
        S_t=S_t, K=K, T=T, r=r, sigma=sigma, q=q, option_type="call"
    )
    put = black_scholes_with_greeks(
        S_t=S_t, K=K, T=T, r=r, sigma=sigma, q=q, option_type="put"
    )

    # put-call parity 0> C - P = S * e^(-q*T) - K*e^(-r*T)
    expected_parity = S_t * np.exp(-q * T) - K * np.exp(-r * T)
    actual_parity = call["Price"] - put["Price"]
    assert abs(actual_parity - expected_parity) < 1e-10

    # delta relationship: call_delta - put_delta = e^(-q*T)
    delta_diff = call["Delta"] - put["Delta"]
    assert abs(delta_diff - np.exp(-q * T)) < 1e-10
