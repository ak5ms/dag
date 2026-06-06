from __future__ import annotations

import argparse
import time
from collections.abc import Callable

import jax
import numpy as np

from trading_dsl_engine.jax_flat import compile_formula
from trading_dsl_engine.jax_flat.engine_cpp import compile_formula as compile_formula_native


CASES = {
    "stateless_chain": "xstd(add(abs(close), div(exp(fraction(open)), add(abs(close), 1.0))))",
    "stateful_shift": "add(cumsum(close), shift(ewm(close, 8.0), lag, 16))",
    "ridge_cat_preds": "get_preds(Ridge(cat(close, open), open, 1.0, 16.0, 0.01))",
    "groupby_cumsum": "groupby((bucket,), close, cumsum(self_))",
    "groupby_nested_rhs": "groupby((bucket,), close, cumsum(cumsum(self_)))",
}


def _time(fn: Callable[[], object], runs: int) -> tuple[float, object]:
    best = float("inf")
    result = None
    for _ in range(runs):
        t0 = time.perf_counter()
        result = fn()
        elapsed = time.perf_counter() - t0
        best = min(best, elapsed)
    return best, result


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare warmed C++ jax_flat tick-loop runtime formulas against warmed JAX-flat batch.")
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--cols", type=int, default=9)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()

    rng = np.random.default_rng(42)
    close = rng.normal(size=(args.rows, args.cols)).astype(np.float64)
    open_ = rng.normal(size=(args.rows, args.cols)).astype(np.float64)
    row = np.arange(args.rows, dtype=np.float64)[:, None]
    col = np.arange(args.cols, dtype=np.float64)[None, :]
    lag = (row + col) % 8.0
    bucket = (row // 32.0 + col) % 128.0
    data = {"close": close, "open": open_, "lag": lag, "bucket": bucket}

    for name, formula in CASES.items():
        # Compile/setup outside timings so reported numbers are steady-state execution only.
        cpp_runtime = compile_formula_native(formula)
        jax_runtime = compile_formula(formula)
        cpp_runtime.run_batch(data)
        jax.block_until_ready(jax_runtime.run_batch(data)[1])

        cpp_s, cpp_result = _time(lambda: cpp_runtime.run_batch(data)[1], args.runs)
        jax_s, jax_result = _time(lambda: jax.block_until_ready(jax_runtime.run_batch(data)[1]), args.runs)
        np.testing.assert_allclose(cpp_result, np.asarray(jax_result), rtol=1e-9, atol=1e-9, equal_nan=True)
        print(
            f"{name}: rows={args.rows} cols={args.cols} "
            f"cpp_tick_best_s={cpp_s:.6f} jax_flat_batch_best_s={jax_s:.6f} "
            f"ratio_cpp_over_jax={cpp_s / jax_s if jax_s else float('nan'):.3f}"
        )


if __name__ == "__main__":
    main()
