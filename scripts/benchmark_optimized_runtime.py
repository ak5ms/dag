from __future__ import annotations

import json
import os
from pathlib import Path
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np

from trading_dsl_engine.base.dsl import add, cumsum, ewm, var
from trading_dsl_engine.jax_flat import compile_formula as compile_current
from trading_dsl_engine.jax_flat.engine import _scan_batch_chunk as current_scan_batch_chunk
from trading_dsl_engine.jax_flat.optimized import (
    _compound_scan_chunk,
    _invalid_like,
    _node_batch_chunk,
    _value_template,
    compile_features,
    compile_formula,
)

jax.config.update("jax_enable_x64", True)

ROWS = int(os.environ.get("BENCH_ROWS", "262144"))
ASSETS = int(os.environ.get("BENCH_ASSETS", "9"))
RUNS = int(os.environ.get("BENCH_RUNS", "5"))
CHUNK = int(os.environ.get("BENCH_CHUNK", "65536"))


def stats(values):
    return {
        "median_s": float(statistics.median(values)),
        "mean_s": float(statistics.mean(values)),
        "min_s": float(min(values)),
        "max_s": float(max(values)),
    }


def timed(fn, warmup=1):
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return stats(samples)


def materialize_tree(value):
    return jax.tree_util.tree_map(lambda x: np.asarray(jax.device_get(x)), value)


def current_run(runtime, data):
    state, out = runtime.run_batch(data)
    materialize_tree((state, out))
    return out


def optimized_run(runtime, data):
    state, out = runtime.run_batch(data, out_path=None)
    materialize_tree(state)
    return out


def hlo_summary_current(runtime, x):
    state = runtime.init_state(ASSETS)
    lowered = current_scan_batch_chunk.lower(runtime, state, (x,), 0)
    compiled = lowered.compile()
    try:
        hlo = lowered.compiler_ir(dialect="hlo").as_hlo_text()
    except Exception:
        hlo = str(lowered.compiler_ir(dialect="stablehlo"))
    memory = compiled.memory_analysis()
    return {
        "while_count": int(hlo.count(" while(")) + int(hlo.count("stablehlo.while")),
        "temp_bytes": int(getattr(memory, "temp_size_in_bytes", 0)),
        "argument_bytes": int(getattr(memory, "argument_size_in_bytes", 0)),
        "output_bytes": int(getattr(memory, "output_size_in_bytes", 0)),
    }


def hlo_summary_optimized(runtime, x):
    inner = runtime.runtime
    state = inner.init_state(ASSETS)
    output_templates = tuple(_value_template(inner.program.nodes[i].op, ASSETS) for i in inner.program.outputs)
    cache_templates = tuple(_value_template(inner.program.nodes[i].op, ASSETS) for i in inner.program.cache_nodes)
    if runtime.execution_strategy() == "compound":
        lowered = _compound_scan_chunk.lower(
            inner,
            state,
            (x,),
            jnp.asarray(x.shape[0], dtype=jnp.int32),
            tuple(_invalid_like(v) for v in output_templates),
            tuple(_invalid_like(v) for v in cache_templates),
        )
    else:
        lowered = _node_batch_chunk.lower(inner, state, (x,), jnp.asarray(0, dtype=jnp.int64))
    compiled = lowered.compile()
    try:
        hlo = lowered.compiler_ir(dialect="hlo").as_hlo_text()
    except Exception:
        hlo = str(lowered.compiler_ir(dialect="stablehlo"))
    memory = compiled.memory_analysis()
    return {
        "strategy": runtime.execution_strategy(),
        "while_count": int(hlo.count(" while(")) + int(hlo.count("stablehlo.while")),
        "temp_bytes": int(getattr(memory, "temp_size_in_bytes", 0)),
        "argument_bytes": int(getattr(memory, "argument_size_in_bytes", 0)),
        "output_bytes": int(getattr(memory, "output_size_in_bytes", 0)),
    }


def ratio(before, after):
    return float(before / after) if after else float("inf")


def main():
    rng = np.random.default_rng(123)
    x_np = rng.normal(size=(ROWS, ASSETS)).astype(np.float64)
    x_np[rng.random(x_np.shape) < 0.01] = np.nan
    x = jnp.asarray(x_np)
    data_jax = {"x": x}
    data_np = {"x": x_np}

    nested = var("x")
    for span in (5.0, 11.0, 23.0, 47.0):
        nested = ewm(nested, span, ignore_na=True, adjust=False)

    current_nested = compile_current(nested, cpp=False)
    optimized_nested = compile_formula(nested, chunk_size=CHUNK, max_in_flight=3)
    current_nested_run = lambda: current_run(current_nested, data_jax)
    optimized_nested_run = lambda: optimized_run(optimized_nested, data_np)
    current_nested_run()
    optimized_nested_run()
    before_nested = timed(current_nested_run)
    after_nested = timed(optimized_nested_run)

    single = ewm(var("x"), 31.0, min_periods=5, ignore_na=True, adjust=False)
    current_single = compile_current(single, cpp=False)
    optimized_single = compile_formula(single, chunk_size=CHUNK, max_in_flight=3)
    current_single_run = lambda: current_run(current_single, data_jax)
    optimized_single_run = lambda: optimized_run(optimized_single, data_np)
    current_single_run()
    optimized_single_run()
    before_single = timed(current_single_run)
    after_single = timed(optimized_single_run)

    formulas = {
        "ewm_5": ewm(var("x"), 5.0, ignore_na=True, adjust=False),
        "ewm_17": ewm(var("x"), 17.0, ignore_na=True, adjust=False),
        "nested": ewm(ewm(var("x"), 5.0, ignore_na=True, adjust=False), 41.0, ignore_na=True, adjust=False),
        "cum": cumsum(add(var("x"), 1.0)),
    }
    current_multi = {name: compile_current(expr, cpp=False) for name, expr in formulas.items()}
    optimized_multi = compile_features(formulas, chunk_size=CHUNK, max_in_flight=3)

    def current_multi_run():
        return {name: current_run(runtime, data_jax) for name, runtime in current_multi.items()}

    def optimized_multi_run():
        return optimized_run(optimized_multi, data_np)

    expected_multi = current_multi_run()
    actual_multi = optimized_multi_run()
    for name in formulas:
        np.testing.assert_allclose(actual_multi[name], np.asarray(expected_multi[name]), rtol=1e-10, atol=1e-10, equal_nan=True)
    before_multi = timed(current_multi_run)
    after_multi = timed(optimized_multi_run)

    x_chunk = x[: min(CHUNK, ROWS)]
    hlo_before = hlo_summary_current(current_nested, x_chunk)
    hlo_after = hlo_summary_optimized(optimized_nested, x_chunk)
    hlo_single_before = hlo_summary_current(current_single, x_chunk)
    hlo_single_after = hlo_summary_optimized(optimized_single, x_chunk)

    result = {
        "environment": {
            "jax_version": jax.__version__,
            "backend": jax.default_backend(),
            "devices": [str(d) for d in jax.devices()],
            "rows": ROWS,
            "assets": ASSETS,
            "runs": RUNS,
            "chunk_size": CHUNK,
        },
        "nested_4x_ewm": {
            "before": before_nested,
            "after": after_nested,
            "speedup_x": ratio(before_nested["median_s"], after_nested["median_s"]),
            "hlo_before": hlo_before,
            "hlo_after": hlo_after,
            "temp_memory_reduction_x": ratio(hlo_before["temp_bytes"], hlo_after["temp_bytes"]),
        },
        "single_affine_ewm": {
            "before": before_single,
            "after": after_single,
            "speedup_x": ratio(before_single["median_s"], after_single["median_s"]),
            "hlo_before": hlo_single_before,
            "hlo_after": hlo_single_after,
            "temp_memory_reduction_x": ratio(hlo_single_before["temp_bytes"], hlo_single_after["temp_bytes"]),
        },
        "four_named_features": {
            "before_separate_runtimes": before_multi,
            "after_shared_runtime": after_multi,
            "speedup_x": ratio(before_multi["median_s"], after_multi["median_s"]),
            "before_total_nodes": sum(len(runtime.program.nodes) for runtime in current_multi.values()),
            "after_shared_nodes": len(optimized_multi.program.nodes),
        },
    }

    Path("benchmark_results.json").write_text(json.dumps(result, indent=2))

    lines = [
        "# JAX-flat optimized runtime benchmark",
        "",
        f"- Backend: `{result['environment']['backend']}`",
        f"- JAX: `{result['environment']['jax_version']}`",
        f"- Input: `{ROWS:,} x {ASSETS}` float64, {RUNS} timed runs",
        f"- Chunk size: `{CHUNK:,}`",
        "",
        "| Case | Before median | After median | Speedup | Before temp | After temp | HLO while before → after |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, label in (("nested_4x_ewm", "4 dependent EWMs"), ("single_affine_ewm", "single affine EWM")):
        case = result[key]
        lines.append(
            f"| {label} | {case['before']['median_s']:.6f}s | {case['after']['median_s']:.6f}s | "
            f"{case['speedup_x']:.2f}x | {case['hlo_before']['temp_bytes']:,} B | "
            f"{case['hlo_after']['temp_bytes']:,} B | {case['hlo_before']['while_count']} → {case['hlo_after']['while_count']} |"
        )
    case = result["four_named_features"]
    lines.extend(
        [
            "",
            "## Multi-root shared DAG",
            "",
            f"- Separate runtimes median: `{case['before_separate_runtimes']['median_s']:.6f}s`",
            f"- Shared runtime median: `{case['after_shared_runtime']['median_s']:.6f}s`",
            f"- Speedup: `{case['speedup_x']:.2f}x`",
            f"- Nodes: `{case['before_total_nodes']}` across separate DAGs → `{case['after_shared_nodes']}` in the shared DAG",
        ]
    )
    Path("benchmark_results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
