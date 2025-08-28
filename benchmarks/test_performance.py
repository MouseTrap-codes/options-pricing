import csv
import logging
from pathlib import Path
from typing import Literal, TypedDict

from models import dp_binomial_tree, recursive_binomial_tree

from .benchmark_utils import bench_time, peak_delta_rss_mib

CSV_FILE = Path("benchmarks/results.csv")


log = logging.getLogger("bench")


class PricerInputs(TypedDict):
    S_t: float
    K: float
    T: float
    r: float
    sigma: float
    q: float
    option_type: Literal["call", "put"]
    N: int


def test_performance_recursive_vs_dp():
    header = [
        "Time Steps (N)",
        "Naive (Recursive) Time (µs)",
        "Naive (Recursive) Peak Memory (MiB)",
        "Optimized (DP + JAX) Time (µs)",
        "Optimized (DP + JAX) Peak Memory (MiB)",
    ]

    with CSV_FILE.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        N_list = [1, 2, 5, 10, 15, 20, 25, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
        for N in N_list:
            hull_inputs: PricerInputs = {
                "S_t": 42.0,
                "K": 40.0,
                "T": 0.25,
                "r": 0.10,
                "sigma": 0.20,
                "q": 0.0,
                "option_type": "call",
                "N": N,
            }

            _ = dp_binomial_tree(**hull_inputs)  # warmup call

            # measure speed
            if N <= 25:
                time_rec = bench_time(lambda: recursive_binomial_tree(**hull_inputs))
            else:
                time_rec = None
            time_dp = bench_time(lambda: dp_binomial_tree(**hull_inputs))

            # measure memory
            if N <= 25:
                mem_rec = peak_delta_rss_mib(
                    lambda: recursive_binomial_tree(**hull_inputs)
                )
            else:
                mem_rec = None
            mem_dp = peak_delta_rss_mib(lambda: dp_binomial_tree(**hull_inputs))

            writer.writerow(
                [
                    N,
                    round(time_rec, 6) if time_rec is not None else "NA",
                    round(mem_rec, 2) if mem_rec is not None else "NA",
                    round(time_dp, 6),
                    round(mem_dp, 2),
                ]
            )

            print(
                f"N={N} | "
                f"Naive (Recursive): {f'{time_rec:.6f}µs' if time_rec is not None else 'NA'}, "
                f"{f'{mem_rec:.2f} MiB' if mem_rec is not None else 'NA'} | "
                f"Optimized (DP + JAX): {time_dp:.6f}µs, {mem_dp:.2f} MiB"
            )


test_performance_recursive_vs_dp()
