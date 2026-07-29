"""Opt-in cpp_new lowering and execution-bridge benchmark.

This benchmark labels the current execution path accurately: until generated
modules are loaded by ``SpecializedRuntime``, cpp_new execution is the generic
flat-native bridge and is not a specialized-kernel performance claim.
"""
from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time

import numpy as np

from trading_dsl_engine.cpp_new import compile_formula as compile_cpp_new
from trading_dsl_engine.jax_flat.engine_cpp import compile_formula as compile_generic


CASES = {
    "ewm_chain": "ewm(ewm(ewm(close, 4.0), 8.0), 16.0)",
    "xs_rank": "xs_rank(close)",
}


def _measure(runtime, data: np.ndarray, samples: int) -> list[float]:
    runtime.run_batch((data,))
    rates = []
    for _ in range(samples):
        state = runtime.init_state(data.shape[1])
        started = time.perf_counter_ns()
        runtime.run_batch((data,), states=state)
        elapsed = (time.perf_counter_ns() - started) * 1e-9
        rates.append(data.shape[0] / elapsed)
    return rates


def benchmark(case: str, rows: int, instruments: int, samples: int) -> dict:
    rng = np.random.default_rng(104729)
    data = rng.normal(size=(rows, instruments))
    data[rng.random(data.shape) < 0.02] = np.nan
    formula = CASES[case]
    started = time.perf_counter_ns()
    generic = compile_generic(formula)
    generic_compile = (time.perf_counter_ns() - started) * 1e-9
    with tempfile.TemporaryDirectory(prefix="cpp_new_benchmark_") as cache:
        started = time.perf_counter_ns()
        specialized = compile_cpp_new(formula, mode="cached-specialized", cache_dir=cache, n_instruments=instruments)
        cold = (time.perf_counter_ns() - started) * 1e-9
        started = time.perf_counter_ns()
        cached = compile_cpp_new(formula, mode="cached-specialized", cache_dir=cache, n_instruments=instruments)
        cached_load = (time.perf_counter_ns() - started) * 1e-9
        old_samples = _measure(generic, data, samples)
        bridge_samples = _measure(cached, data, samples)
        old_output = generic.run_batch((data,))[1]
        new_output = cached.run_batch((data,))[1]
        np.testing.assert_allclose(old_output, new_output, equal_nan=True)
        source_bytes = specialized.artifact.source.stat().st_size
    return {
        "case": case, "rows": rows, "instruments": instruments,
        "generic_compile_seconds": generic_compile,
        "cpp_new_cold_materialization_seconds": cold,
        "cpp_new_cached_materialization_seconds": cached_load,
        "generated_source_bytes": source_bytes,
        "generic_rows_per_second_samples": old_samples,
        "generic_rows_per_second_median": statistics.median(old_samples),
        "cpp_new_generic_bridge_rows_per_second_samples": bridge_samples,
        "cpp_new_generic_bridge_rows_per_second_median": statistics.median(bridge_samples),
        "execution_tier": "generic-flat-native-bridge",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=CASES, default="ewm_chain")
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--instruments", type=int, default=150)
    parser.add_argument("--samples", type=int, default=5)
    print(json.dumps(benchmark(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
