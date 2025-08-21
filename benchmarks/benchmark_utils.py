import os
import threading
import time
from typing import Any, Callable

import psutil


def _sync_if_jax(x: Any) -> None:
    if hasattr(x, "block_until_ready"):
        x.block_until_ready()


def peak_delta_rss_mib(fn: Callable[[], Any], *, interval: float = 0.0005) -> float:
    proc = psutil.Process(os.getpid())
    stop = threading.Event()
    baseline = peak = 0

    def watcher():
        nonlocal baseline, peak
        first = True
        while not stop.is_set():
            rss = proc.memory_info().rss
            if first:
                baseline = peak = rss
                first = False
            elif rss > peak:
                peak = rss
            time.sleep(interval)

    t = threading.Thread(target=watcher, daemon=True)
    t.start()
    out = fn()
    _sync_if_jax(out)
    stop.set()
    t.join()
    return max(0.0, (peak - baseline) / (1024**2))


def bench_time(fn: Callable[[], Any], runs: int = 10, warmups: int = 2) -> float:
    for _ in range(warmups):
        _sync_if_jax(fn())
    best = float("inf")
    for _ in range(runs):
        t0 = time.perf_counter_ns()
        _sync_if_jax(fn())
        dt_us = (time.perf_counter_ns() - t0) / 1_000.0
        if dt_us < best:
            best = dt_us
    return best
