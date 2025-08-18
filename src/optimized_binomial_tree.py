from typing import Callable, Optional, cast

import jax # type: ignore
import jax.numpy as jnp # type: ignore
import numpy as np

from binomial_tree_utils import compute_u, compute_d, compute_probability

def _dp_kernel_py(
    S_t: float,
    K: float,
    N: int,
    u: float,
    d: float,
    p: float,
    disc: float,
    is_call: int,
    is_amer: int,
) -> jnp.ndarray:
    j = jnp.arange(N + 1, dtype=jnp.float64)

    # stable stock prices at maturity: S = S0 * d^N * (u/d)^j
    S = S_t * (d**N) * ((u / d) ** j)

    V_call = jnp.maximum(0.0, S - K)
    V_put = jnp.maximum(K - S, 0.0)
    V = jnp.where(is_call == 1, V_call, V_put)

    # like Bellman equations in RL!
    def bellman_like_equation(
        i: int, carry: tuple[jnp.ndarray, jnp.ndarray]
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        V_layer, S_layer = carry
        k = N - i
        # continuation (for european)
        cont = disc * (p * V_layer[1 : k + 1] + (1.0 - p) * V_layer[:k])

        # move underlying prices back one layer (for american)
        S_prev = S_layer[:k] / d

        # intrinsic for current layer
        intrinsic = jnp.where(is_call == 1, S_prev - K, K - S_prev)
        intrinsic = jnp.maximum(intrinsic, 0.0)

        # american vs european
        new_V = jnp.where(is_amer == 1, jnp.maximum(cont, intrinsic), cont)

        # keep shapes invariant for jax to not bug out
        V_layer = V_layer.at[:k].set(new_V)
        S_layer = S_layer.at[:k].set(S_prev)

        return (V_layer, S_layer)

    V_final, _ = jax.lax.fori_loop(0, N, bellman_like_equation, (V, S))

    return V_final[0]


_dp_kernel: Callable[..., jnp.ndarray] = cast(
    Callable[..., jnp.ndarray], jax.jit(_dp_kernel_py)
)


# binomial tree using dp
def dp_binomial_tree(
    sigma: float,
    T: float,
    N: int,
    S_t: float,
    r: float,
    K: float,
    q: Optional[float] = None,
    r_f: Optional[float] = None,
    asset_type: str = "nondividend",
    exercise_type: str = "european",
    option_type: str = "call",
) -> float:
    if N <= 0:
        raise ValueError("Cannot have 0 or lower time periods")

    dt = T / N
    u = compute_u(sigma, dt)
    d = compute_d(sigma, dt)
    p = compute_probability(u=u, d=d, r=r, dt=dt, q=q, r_f=r_f, asset_type=asset_type)

    discount_rate = np.exp(-r * dt)

    is_call = 1 if option_type.lower() == "call" else 0
    is_amer = 1 if exercise_type.lower() == "american" else 0

    return float(
        _dp_kernel(
            float(S_t),
            float(K),
            int(N),
            float(u),
            float(d),
            float(p),
            float(discount_rate),
            int(is_call),
            int(is_amer),
        )
    )