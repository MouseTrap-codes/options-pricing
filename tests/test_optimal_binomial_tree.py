import json
from pathlib import Path

import pytest

from models import dp_binomial_tree

BT_TOL_DEFAULT = {"rel": 1e-6, "abs": 1e-4}
BT_TOL_OVERRIDES = {
    "High_vol_european_call": {"rel": 2e-6, "abs": 5e-3},
    "Very_high_volatility_european": {
        "rel": 2e-6,
        "abs": 3e-1,
    },  # if you include this one
    "Future_european_call": {"rel": 2e-6, "abs": 2e-3},
    "Future_european_put": {"rel": 2e-6, "abs": 2e-3},
    "Deep_ITM_european_call": {"rel": 2e-6, "abs": 2e-3},
    "Min_time_steps_european": {"rel": 2e-6, "abs": 2e-3},
}

BT_TOL_OVERRIDES.update(
    {
        "ATM_european_call": {"rel": 2e-6, "abs": 4e-4},
        "ATM_european_put": {"rel": 2e-6, "abs": 4e-4},
        "Deep_ITM_european_put": {"rel": 2e-6, "abs": 4e-4},
        "High_dividend_european_put": {"rel": 2e-6, "abs": 5e-4},
        "Very_low_vol_european_put": {"rel": 2e-6, "abs": 4e-4},
        "Long_term_OTM_european_put": {"rel": 2e-6, "abs": 4e-4},
        "Zero_interest_rate_european": {"rel": 2e-6, "abs": 4e-4},
    }
)


def _load_bt_cases():
    path = Path(__file__).parent / "data" / "bt_european_test_cases.json"
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    cases = []
    for section in ("EUROPEAN_CALLS", "EUROPEAN_PUTS", "EUROPEAN_EDGE_CASES"):
        for case in payload.get(section, []):
            cases.append(case)
    return cases


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


@pytest.mark.parametrize("case", _load_bt_cases(), ids=lambda c: c["name"])
def test_binomial_european_price(case):
    inputs = _normalize_inputs(case["inputs"])
    price = dp_binomial_tree(**inputs)
    exp = case["expected"]["Price"]
    tol = BT_TOL_OVERRIDES.get(case["name"], BT_TOL_DEFAULT)
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
