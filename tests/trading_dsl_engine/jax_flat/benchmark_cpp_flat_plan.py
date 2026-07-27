"""Opt-in native-plan baseline benchmark and Linux profiler entry point.

Run directly; this is deliberately not collected as a correctness test::

    python tests/trading_dsl_engine/jax_flat/benchmark_cpp_flat_plan.py --rows 4096 --instruments 150
    perf stat -e cycles,instructions,cache-misses,branch-misses -- python \
      tests/trading_dsl_engine/jax_flat/benchmark_cpp_flat_plan.py --case elementwise
    perf record -g -- python tests/trading_dsl_engine/jax_flat/benchmark_cpp_flat_plan.py --case groupby

The allocation figure covers Python-visible allocations. Native zero-allocation
proof will be added with the arena stage; it must not be inferred from this
baseline measurement.
"""

from __future__ import annotations

import argparse
import json
import resource
import time
import tracemalloc

import numpy as np

from trading_dsl_engine.jax_flat.engine_cpp import compile_formula


CASES = {
    "elementwise": "add(mul(add(mul(add(mul(close, 1.01), open), 0.99), close), 1.001), open)",
    "rank_stateful": "xs_rank(cumsum(close))",
    "ewm_chain": "ewm(ewm(ewm(close, 4.0), 8.0), 16.0)",
    "rolling_shift": "shift(roll_mean(close, 16), lag, 32)",
    "groupby": "groupby((key0, key1), close, add(cumsum(self_), 1.0))",
    "groupby_locality": "groupby((key0, key1), close, add(cumsum(self_), 1.0))",
    "groupby_churn": "groupby((key0, key1), close, add(cumsum(self_), 1.0))",
    "universe_groupby": "groupby((univ([0, 1], [2, 3]), key0), close, cumsum(self_))",
    "ridge": "get_preds(Ridge(cat(close, open), open, 1.0, 16.0, 0.01))",
}


def benchmark(case: str, rows: int, instruments: int, ticks: int, runs: int = 1) -> dict[str, object]:
    rng = np.random.default_rng(7)
    data = {
        "close": rng.normal(size=(rows, instruments)),
        "open": rng.normal(size=(rows, instruments)),
        "lag": rng.integers(0, 16, size=(rows, instruments)).astype(np.float64),
        "key0": rng.integers(0, 16, size=(rows, instruments)).astype(np.float64),
        "key1": rng.integers(0, 4, size=(rows, instruments)).astype(np.float64),
    }
    formula = CASES[case]
    if case == "groupby_locality":
        data["key0"].fill(0.0)
        data["key1"].fill(0.0)
    elif case == "groupby_churn":
        instrument = np.arange(instruments, dtype=np.float64)[None, :]
        timestep = np.arange(rows, dtype=np.float64)[:, None]
        data["key0"] = np.mod(instrument + timestep, 128.0)
        data["key1"] = np.mod(3.0 * instrument + timestep, 8.0)
    if case == "universe_groupby" and instruments < 4:
        raise ValueError("universe_groupby requires at least four instruments")
    t0 = time.perf_counter()
    runtime = compile_formula(formula)
    state = runtime.init_state(instruments)
    cold_seconds = time.perf_counter() - t0

    inputs = tuple(data[name][0] for name in runtime.program.input_names)
    out = np.empty(instruments, dtype=np.float64)
    runtime.tick_into(state, out, *inputs)  # warm construction/lazy paths
    tick_rates = []
    batch_rates = []
    for _ in range(runs):
        state = runtime.init_state(instruments)
        t0 = time.perf_counter()
        for _ in range(ticks):
            runtime.tick_into(state, out, *inputs)
        tick_rates.append(ticks / (time.perf_counter() - t0))

    allocation_state = runtime.init_state(instruments)
    tracemalloc.start()
    before = tracemalloc.take_snapshot()
    for _ in range(min(ticks, 100)):
        runtime.tick_into(allocation_state, out, *inputs)
    after = tracemalloc.take_snapshot()
    allocations = sum(max(0, item.count_diff) for item in after.compare_to(before, "lineno"))
    tracemalloc.stop()

    batch_inputs = {name: data[name] for name in runtime.program.input_names}
    for _ in range(runs):
        state = runtime.init_state(instruments)
        t0 = time.perf_counter()
        runtime.run_batch(batch_inputs, states=state)
        batch_rates.append(rows / (time.perf_counter() - t0))
    return {
        "case": case,
        "nodes": len(runtime.native_plan.nodes),
        "rows": rows,
        "instruments": instruments,
        "cold_seconds": cold_seconds,
        "runs": runs,
        "ticks_per_second": float(np.median(tick_rates)),
        "batch_rows_per_second": float(np.median(batch_rates)),
        "tick_samples": tick_rates,
        "batch_samples": batch_rates,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "output_bytes_per_tick": out.nbytes,
        "python_allocation_count": allocations,
        "frontier_transfer_seconds": 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=tuple(CASES) + ("all",), default="all")
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--instruments", type=int, choices=(150, 1000, 4000), default=150)
    parser.add_argument("--ticks", type=int, default=2000)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()
    cases = CASES if args.case == "all" else (args.case,)
    for case in cases:
        print(json.dumps(benchmark(case, args.rows, args.instruments, args.ticks, args.runs), sort_keys=True))


if __name__ == "__main__":
    main()
