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
from collections import Counter
import json
import os
import resource
import tempfile
import time
import tracemalloc

import numpy as np

from trading_dsl_engine.jax_flat.engine_cpp import compile_formula


def _alpha_sharpes_formula():
    from flows.alpha_search import default_alpha_pnl
    from flows.utils import pct_change
    from trading_dsl_engine.base.dsl import cat, ewm, var, xs_rank

    returns = pct_change(var("mp_out0.close"))
    features = [xs_rank(ewm(returns, span)) for span in range(1, 30)]
    return cat(
        *(
            default_alpha_pnl(
                feature,
                roll_rets=returns,
                is_tradable=var("is_tradable_out0"),
                hl=1440,
            )
            for feature in features
        )
    )


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
    "wide_frontier": "add(add(exp(close), ln(abs(open))), add(xs_rank(high), xs_rank(low)))",
    "alpha_sharpes": None,
}


def benchmark_alpha_sharpes(rows: int, instruments: int, runs: int, *, cpp: bool) -> dict[str, object]:
    from trading_dsl_engine.jax_flat.engine import compile_formula as compile_formula_auto
    from trading_dsl_engine.jax_flat.engine_cpp import lower_native_plan

    rng = np.random.default_rng(19)
    returns = rng.normal(scale=1e-3, size=(rows, instruments))
    mid = 100.0 * np.exp(np.cumsum(returns, axis=0))
    inputs = {
        "mp_out0.close": mid,
        "is_tradable_out0": (rng.random((rows, instruments)) > 0.02).astype(np.float64),
    }
    t0 = time.perf_counter()
    runtime = compile_formula_auto(_alpha_sharpes_formula(), cpp=cpp)
    construction_seconds = time.perf_counter() - t0
    plan, _ = lower_native_plan(runtime.program)

    samples = []
    output_bytes = rows * instruments * 29 * np.dtype(np.float64).itemsize
    for run in range(runs + 1):
        fd, path = tempfile.mkstemp(prefix="alpha_sharpes_", suffix=".memmap")
        os.close(fd)
        try:
            t0 = time.perf_counter()
            _, out = runtime.run_batch(inputs, out_path=path)
            elapsed = time.perf_counter() - t0
            if run:
                samples.append(rows / elapsed)
            del out
        finally:
            os.unlink(path)
    return {
        "case": "alpha_sharpes",
        "backend": "cpp" if cpp else "jax",
        "rows": rows,
        "instruments": instruments,
        "source_nodes": len(runtime.program.nodes),
        "optimized_native_nodes": len(plan.nodes),
        "optimizations": dict(plan.optimizations),
        "opcodes": dict(Counter(node.opcode for node in plan.nodes)),
        "construction_seconds": construction_seconds,
        "output_bytes": output_bytes,
        "runs": runs,
        "batch_rows_per_second": float(np.median(samples)),
        "batch_samples": samples,
    }


def benchmark(case: str, rows: int, instruments: int, ticks: int, runs: int = 1, *, workers: int | None = None) -> dict[str, object]:
    rng = np.random.default_rng(7)
    data = {
        "close": rng.normal(size=(rows, instruments)),
        "open": rng.normal(size=(rows, instruments)),
        "high": rng.normal(size=(rows, instruments)),
        "low": rng.normal(size=(rows, instruments)),
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
    runtime = compile_formula(formula, workers=workers)
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
        "workers": runtime.workers,
        "graph_shape": "wide" if case == "wide_frontier" else "chain_or_specialized",
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
    parser.add_argument("--backend", choices=("cpp", "jax"), default="cpp")
    parser.add_argument("--workers", default="1,2,4", help="comma-separated native worker counts")
    args = parser.parse_args()
    cases = CASES if args.case == "all" else (args.case,)
    for case in cases:
        worker_values = [None] if args.backend == "jax" or case == "alpha_sharpes" else [int(v) for v in args.workers.split(",")]
        baseline = None
        for workers in worker_values:
            result = (
            benchmark_alpha_sharpes(
                args.rows, args.instruments, args.runs, cpp=args.backend == "cpp"
            )
            if case == "alpha_sharpes"
                else benchmark(case, args.rows, args.instruments, args.ticks, args.runs, workers=workers)
            )
            rate = result["batch_rows_per_second"]
            baseline = rate if baseline is None else baseline
            result["speedup"] = rate / baseline
            print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
