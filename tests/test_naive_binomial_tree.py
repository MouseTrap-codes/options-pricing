import numpy as np
import pytest

from models import dp_binomial_tree, recursive_binomial_tree


@pytest.mark.parametrize(
    "inputs",
    [
        dict(
            S_t=100,
            K=100,
            T=1,
            r=0.05,
            sigma=0.2,
            N=1,
            q=0.0,
            r_f=None,
            asset_type="nondividend",
            exercise_type="european",
            option_type="call",
        ),
        dict(
            S_t=100,
            K=110,
            T=1,
            r=0.05,
            sigma=0.2,
            N=1,
            q=0.0,
            r_f=None,
            asset_type="nondividend",
            exercise_type="european",
            option_type="put",
        ),
        dict(
            S_t=100,
            K=105,
            T=1,
            r=0.05,
            sigma=0.2,
            N=2,
            q=0.0,
            r_f=None,
            asset_type="nondividend",
            exercise_type="european",
            option_type="put",
        ),
    ],
)
def test_recursive_vs_dp_european(inputs):
    expected = dp_binomial_tree(**inputs)
    actual = recursive_binomial_tree(**inputs)
    assert actual == pytest.approx(expected, rel=1e-3, abs=1e-3)


@pytest.mark.parametrize(
    "inputs",
    [
        dict(
            S_t=100,
            K=100,
            T=1,
            r=0.05,
            sigma=0.2,
            N=1,
            q=0.0,
            r_f=None,
            asset_type="nondividend",
            exercise_type="american",
            option_type="call",
        ),
        dict(
            S_t=100,
            K=110,
            T=1,
            r=0.05,
            sigma=0.2,
            N=1,
            q=0.0,
            r_f=None,
            asset_type="nondividend",
            exercise_type="american",
            option_type="put",
        ),
        dict(
            S_t=100,
            K=105,
            T=1,
            r=0.05,
            sigma=0.2,
            N=2,
            q=0.0,
            r_f=None,
            asset_type="nondividend",
            exercise_type="american",
            option_type="put",
        ),
    ],
)
def test_recursive_vs_dp_american(inputs):
    expected = dp_binomial_tree(**inputs)
    actual = recursive_binomial_tree(**inputs)
    assert actual == pytest.approx(expected, rel=1e-3, abs=1e-3)


def test_put_call_parity():
    S_t = 42.0
    K = 40.0
    T = 0.25
    r = 0.10
    sigma = 0.20
    q = 0.0
    N = 4

    call = recursive_binomial_tree(
        S_t=S_t,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
        q=q,
        r_f=None,
        asset_type="nondividend",
        option_type="call",
        exercise_type="european",
        N=N,
    )
    put = recursive_binomial_tree(
        S_t=S_t,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
        q=q,
        r_f=None,
        asset_type="nondividend",
        option_type="put",
        exercise_type="european",
        N=N,
    )

    expected = S_t * np.exp(-q * T) - K * np.exp(-r * T)
    assert abs((call - put) - expected) < 1e-3


@pytest.mark.parametrize(
    "inputs",
    [
        dict(S_t=-100, K=100, T=1, r=0.05, sigma=0.2, N=2, option_type="call"),
        dict(S_t=100, K=-100, T=1, r=0.05, sigma=0.2, N=2, option_type="call"),
        dict(S_t=100, K=100, T=1, r=0.05, sigma=0.2, N=0, option_type="call"),
        dict(S_t=100, K=100, T=1, r=0.05, sigma=0.2, N=2, option_type="banana"),
    ],
)
def test_invalid_inputs(inputs):
    with pytest.raises(ValueError):
        recursive_binomial_tree(**inputs)
