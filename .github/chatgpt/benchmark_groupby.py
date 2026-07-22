from __future__ import annotations

import argparse
import gc
import json
import os
import time
import warnings
from pathlib import Path

import jax
import numpy as np

from trading_dsl_engine.base.dsl import cumsum, groupby, self_, var
from trading_dsl_engine.jax_flat import compile_formula, stateless

jax.config.update("jax_enable_x64", True)


def _block(value):
    return jax.tree_util.tree_map(
        lambda leaf: leaf.block_until_ready() if hasattr(leaf, "block_until_ready") else leaf,
        value,
    )


def _data(rows: int, cols: int = 9) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(20260722)
    row = np.arange(rows, dtype=np.float64)[:, None]
    col = np.arange(cols, dtype=np.float64)[None, :]
    x = rng.normal(size=(rows, cols)).astype(np.float64)
    y = (0.25 * rng.normal(size=(rows, cols)) + np.sin(row * 0.001 + col)).astype(np.float64)
    key = np.mod(row // 32.0 + np.mod(col, 3.0), 128.0)
    x[::997, 2] = np.nan
    y[::1231, 7] = np.nan
    return {"key": key, "x": x, "y": y}


def _bench(name, formula, *, rows: int, cpp: bool, runs: int, warmups: int = 1):
    data = _data(rows)
    runtime = compile_formula(formula, cpp=cpp)
    state = out = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for _ in range(warmups):
            state, out = runtime.run_batch(data)
            _block(out)
        elapsed = []
        for _ in range(runs):
            start = time.perf_counter()
            state, out = runtime.run_batch(data)
            _block(out)
            elapsed.append(time.perf_counter() - start)
    array = np.asarray(out)
    result = {
        "name": name,
        "rows": rows,
        "cols": int(array.shape[1]),
        "cpp_requested": cpp,
        "state_type": type(state).__name__,
        "runs_s": elapsed,
        "median_s": float(np.median(elapsed)),
        "elements_per_s": float(rows * array.shape[1] / np.median(elapsed)),
        "checksum": float(np.nansum(array)),
    }
    del runtime, data, state, out, array
    gc.collect()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("baseline", "optimized"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    nested = "groupby((key,), x, cumsum(ewm(add(self_, y), 8.0)))"
    custom_kwargs = {"group_memberwise": True} if args.variant == "optimized" else {}
    custom = stateless(lambda x, y: x + 2.0 * y, name="custom_affine", **custom_kwargs)
    custom_formula = groupby((var("key"),), var("x"), cumsum(custom(self_, var("y"))))

    results = {
        "variant": args.variant,
        "jax_version": jax.__version__,
        "platform": jax.default_backend(),
        "cpu_count": os.cpu_count(),
        "benchmarks": [],
    }
    results["benchmarks"].append(
        _bench("pure_jax_nested_10k", nested, rows=10_000, cpp=False, runs=3)
    )
    results["benchmarks"].append(
        _bench("custom_groupby_10k", custom_formula, rows=10_000, cpp=False, runs=3)
    )
    results["benchmarks"].append(
        _bench("default_multifeed_10k", nested, rows=10_000, cpp=True, runs=3)
    )

    if args.variant == "optimized":
        results["benchmarks"].append(
            _bench("pure_jax_nested_100k", nested, rows=100_000, cpp=False, runs=2)
        )
        results["benchmarks"].append(
            _bench("custom_groupby_100k", custom_formula, rows=100_000, cpp=False, runs=2)
        )
        results["benchmarks"].append(
            _bench("native_multifeed_1m", nested, rows=1_000_000, cpp=True, runs=3)
        )

    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
