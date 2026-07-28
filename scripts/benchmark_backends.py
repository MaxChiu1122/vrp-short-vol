"""Benchmark the pure-Python VRP core against mcpricer's C++ engine.

Usage:  python scripts/benchmark_backends.py

Times the identical experiment through both backends and reports the speedup, plus
a distributional check that they agree (they draw from different generators, so the
comparison is statistical, never elementwise).
"""

from __future__ import annotations

import time

import numpy as np

from vrp import DEFAULT_CONTRACT, summarize_pnl, vrp_pnl_paths

SIGMA_IMPLIED = 0.20
SIGMA_REALIZED = 0.16
SEED = 11
PATH_COUNTS = (10_000, 50_000, 200_000)


def _time(fn, *, repeats: int = 3) -> tuple[float, np.ndarray]:
    """Best-of-N wall time; the minimum is the least noisy estimator here."""
    best = float("inf")
    out = None
    for _ in range(repeats):
        start = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - start)
    return best, out


def main() -> None:
    print(f"contract: {DEFAULT_CONTRACT}")
    print(f"selling {SIGMA_IMPLIED:.0%} into {SIGMA_REALIZED:.0%} realized\n")

    def run(backend: str, n_paths: int):
        return lambda: vrp_pnl_paths(
            sigma_implied=SIGMA_IMPLIED,
            sigma_realized=SIGMA_REALIZED,
            n_paths=n_paths,
            seed=SEED,
            backend=backend,
        )

    print(f"{'paths':>10}  {'python (s)':>11}  {'fast (s)':>10}  {'speedup':>8}  {'paths/s (fast)':>15}")
    print("-" * 62)
    for n_paths in PATH_COUNTS:
        t_py, pnl_py = _time(run("python", n_paths))
        t_fast, pnl_fast = _time(run("fast", n_paths))
        print(
            f"{n_paths:>10,}  {t_py:>11.3f}  {t_fast:>10.4f}  {t_py / t_fast:>7.0f}x  "
            f"{n_paths / t_fast:>15,.0f}"
        )

    # Do the two engines actually agree? Compare the largest run's distributions.
    se = np.sqrt(pnl_py.var(ddof=1) / pnl_py.size + pnl_fast.var(ddof=1) / pnl_fast.size)
    gap = abs(pnl_py.mean() - pnl_fast.mean())
    print("\ndistributional agreement at the largest path count:")
    print(f"  python: {summarize_pnl(pnl_py)}")
    print(f"  fast  : {summarize_pnl(pnl_fast)}")
    print(f"  mean gap {gap:+.5f} = {gap / se:.2f} combined s.e.  (expect < ~2)")


if __name__ == "__main__":
    main()
