from typing import Optional

import numpy as np


def compute_u(sigma: float, dt: float) -> float:
    return float(np.exp(sigma * np.sqrt(dt)))


def compute_d(sigma: float, dt: float) -> float:
    return float(1.0 / compute_u(sigma=sigma, dt=dt))


def compute_probability(
    u: float,
    d: float,
    r: float,
    dt: float,
    q: Optional[float] = None,
    r_f: Optional[float] = None,
    asset_type: str = "nondividend",
) -> float:
    if u < d:
        raise ValueError(f"Need u > d (got u={u}, d={d})")
    if u - d == 0:
        raise ZeroDivisionError("u - d == 0")

    asset_type = asset_type.lower()

    def nondividend() -> float:
        return float(np.exp(r * dt))

    def dividend() -> float:
        if q is None:
            raise ValueError("Need q")
        return float(np.exp((r - q) * dt))

    def currency() -> float:
        if r_f is None:
            raise ValueError("Foreign risk-free rate not specified.")
        return float(np.exp((r - r_f) * dt))

    def future() -> float:
        return 1.0

    computations = {
        "nondividend": nondividend,
        "dividend": dividend,
        "currency": currency,
        "future": future,
    }

    if asset_type not in computations:
        raise ValueError(f"Unknown asset type: '{asset_type}'")

    a: float = computations[asset_type]()

    p = (a - d) / (u - d)

    if not (0.0 <= p <= 1.0):
        raise ValueError(
            f"Risk-neutral probability out of bounds: p={p:.6f}. Check inputs."
        )

    return p
