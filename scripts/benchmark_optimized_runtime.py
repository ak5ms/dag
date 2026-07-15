from __future__ import annotations

import json
import os
from pathlib import Path
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np

from trading_dsl_engine.base.dsl import ewm, var
from trading_dsl_engine.jax_flat import compile_formula as compile_current
from trading_dsl_engine.jax_flat import compile_features, compile_optimized_formula

jax.config.update("jax_enable_x64", True)

ROWS = int(os.environ.get("BENCH_ROWS", "262144"))
ASSETS = int(os.environ.get("BENCH_ASSETS", "9"))
RUNS = int(os.environ.get("BENCH_RUNS", "5"))


def _stats(values):
    return {
        "median_s": float(statistics.median(values)),
        "mean_s": float(statistics.mean(values)),
        "min_s": float(min(values)),
        "max_s": float(max(values)),
    }


def _time(fn):
    fn()  # compile and warm; excluded
    samples = []
    for _ in range(RUNS):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return _stats(samples)


def _materialize(value):
    jax.tree_util.tree_map(lambda x: np.asarray(jax.device_get(x)), value)


def _run_current(runtime, data):
    result = runtime.run_batch(data)
    _materialize(result)


def _run_optimized(runtime, data):
    result = runtime.run_batch(data, out_path=None)
    _materialize(result)


def _ratio(before, after):
    return float(before / after)


def main():
    rng = np.random.default_rng(123)
    x_np = rng.normal(size=(ROWS, ASSETS)).astype(np.float64)
    x_np[rng.random(x_np.shape) < 0.01] = np.nan
    data_np = {"x": x_np}
    data_jax = {"x": jnp.asarray(x_np)}

    x = var("x")
    nested = x
    for span in (5.0, 11.0, 23.0, 47.0, 71.0, 103.0, 151.0, 211.0):
        nested = ewm(nested, span, ignore_na=True, adjust=False)

    current_nested = compile_current(nested, cpp=False)
    optimized_nested = compile_optimized_formula(nested)
    before_nested = _time(lambda: _run_current(current_nested, data_jax))
    after_nested = _time(lambda: _run_optimized(optimized_nested, data_np))

    shared = ewm(x, 7.0, ignore_na=True, adjust=False)
    formulas = {
        "fast": ewm(ewm(shared, 11.0, ignore_na=True, adjust=False), 19.0, ignore_na=True, adjust=False),
        "medium": ewm(ewm(shared, 13.0, ignore_na=True, adjust=False), 23.0, ignore_na=True, adjust=False),
        "slow": ewm(ewm(shared, 17.0, ignore_na=True, adjust=False), 31.0, ignore_na=True, adjust=False),
        "slower": ewm(ewm(shared, 21.0, ignore_na=True, adjust=False), 43.0, ignore_na=True, adjust=False),
    }
    current_branches = {name: compile_current(expr, cpp=False) for name, expr in formulas.items()}
    optimized_branches = compile_features(formulas)

    def run_current_branches():
        for runtime in current_branches.values():
            _run_current(runtime, data_jax)

    before_branches = _time(run_current_branches)
    after_branches = _time(lambda: _run_optimized(optimized_branches, data_np))

    result = {
        "environment": {
            "jax_version": jax.__version__,
            "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
            "rows": ROWS,
            "assets": ASSETS,
            "runs": RUNS,
        },
        "nested_8x_ewm": {
            "before": before_nested,
            "after": after_nested,
            "speedup_x": _ratio(before_nested["median_s"], after_nested["median_s"]),
            "strategy": optimized_nested.execution_strategy(),
        },
        "four_ewm_branches": {
            "before": before_branches,
            "after": after_branches,
            "speedup_x": _ratio(before_branches["median_s"], after_branches["median_s"]),
            "strategy": optimized_branches.execution_strategy(),
        },
    }

    Path("benchmark_results.json").write_text(json.dumps(result, indent=2))
    lines = [
        "# Optimized JAX-flat CPU benchmark",
        "",
        f"- JAX: `{jax.__version__}`",
        f"- Input: `{ROWS:,} x {ASSETS}` float64",
        f"- Timed runs: `{RUNS}` after one excluded warmup",
        "",
        "| Case | Before | After | Speedup | Strategy |",
        "|---|---:|---:|---:|---|",
    ]
    for key, label in (("nested_8x_ewm", "8 dependent EWMs"), ("four_ewm_branches", "4 shared-prefix EWM branches")):
        case = result[key]
        lines.append(
            f"| {label} | {case['before']['median_s']:.6f}s | {case['after']['median_s']:.6f}s | "
            f"{case['speedup_x']:.2f}x | `{case['strategy']}` |"
        )
    Path("benchmark_results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
