"""Opt-in cpp_new lowering and native lane-family benchmark.

The EWM ``cat`` case uses the fused native parameter-lane family. Other cases
remain explicitly labelled as the generic flat-native bridge.
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
    "cat_ewm": None,
    "cat_rank_ewm": None,
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


def _measure_into(runtime, data: np.ndarray, lanes: int, samples: int) -> list[float]:
    output = np.empty((data.shape[0], data.shape[1], lanes), dtype=np.float64)
    rates = []
    for _ in range(samples):
        state = runtime.init_state(data.shape[1])
        started = time.perf_counter_ns()
        runtime.run_batch((data,), states=state, out=output)
        elapsed = (time.perf_counter_ns() - started) * 1e-9
        rates.append(data.shape[0] / elapsed)
    return rates


def _ablate(runtime, data: np.ndarray, lanes: int, samples: int) -> dict[str, dict[str, object]]:
    variants = ("lane-major", "instrument-major", "materialized", "store-only")
    output = np.empty((data.shape[0], data.shape[1], lanes), dtype=np.float64)
    results = {}
    for variant in variants:
        rates = []
        for _ in range(samples):
            state = runtime.init_state(data.shape[1])
            started = time.perf_counter_ns()
            runtime.run_batch_ablation(state, output, (data,), variant)
            elapsed = (time.perf_counter_ns() - started) * 1e-9
            rates.append(data.shape[0] / elapsed)
        median = statistics.median(rates)
        results[variant] = {
            "rows_per_second_samples": rates,
            "rows_per_second_median": median,
            "output_gib_per_second": median * data.shape[1] * lanes * 8 / 2**30,
        }
    return results


def benchmark(case: str, rows: int, instruments: int, samples: int, lanes: int = 16) -> dict:
    rng = np.random.default_rng(104729)
    data = rng.normal(size=(rows, instruments))
    data[rng.random(data.shape) < 0.02] = np.nan
    formula = CASES[case]
    if case in {"cat_ewm", "cat_rank_ewm"}:
        spans = (2.0 ** (1.0 + np.arange(lanes) / 4.0)).tolist()
        branch = (lambda span: f"ewm(close, {span!r})") if case == "cat_ewm" else (lambda span: f"xs_rank(ewm(close, {span!r}))")
        formula = "cat(" + ",".join(branch(span) for span in spans) + ")"
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
        old_into = _measure_into(generic, data, lanes, samples) if case == "cat_ewm" else None
        cpp_new_into = _measure_into(cached, data, lanes, samples) if case == "cat_ewm" else None
        old_output = generic.run_batch((data,))[1]
        new_output = cached.run_batch((data,))[1]
        np.testing.assert_allclose(old_output, new_output, equal_nan=True)
        source_bytes = specialized.artifact.source.stat().st_size
        ablations = _ablate(cached, data, lanes, samples) if case == "cat_ewm" else None
    return {
        "case": case, "rows": rows, "instruments": instruments, "lanes": lanes if case == "cat_ewm" else None,
        "generic_compile_seconds": generic_compile,
        "cpp_new_cold_materialization_seconds": cold,
        "cpp_new_cached_materialization_seconds": cached_load,
        "generated_source_bytes": source_bytes,
        "generic_rows_per_second_samples": old_samples,
        "generic_rows_per_second_median": statistics.median(old_samples),
        "cpp_new_rows_per_second_samples": bridge_samples,
        "cpp_new_rows_per_second_median": statistics.median(bridge_samples),
        "execution_tier": cached.execution_tier,
        "generic_direct_output_rows_per_second_samples": old_into,
        "generic_direct_output_rows_per_second_median": statistics.median(old_into) if old_into else None,
        "cpp_new_direct_output_rows_per_second_samples": cpp_new_into,
        "cpp_new_direct_output_rows_per_second_median": statistics.median(cpp_new_into) if cpp_new_into else None,
        "ablations": ablations,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=CASES, default="ewm_chain")
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--instruments", type=int, default=150)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--lanes", type=int, default=16)
    print(json.dumps(benchmark(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
