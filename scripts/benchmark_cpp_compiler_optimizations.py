"""Benchmark compiler-only cpp_stream optimizations without changing operators."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

import numpy as np

from trading_dsl_engine.base.dsl import (
    ewm,
    mean,
    rolling_mean,
    rolling_std,
    rolling_sum,
    shift,
    std,
    var,
    xs_rank,
)
from trading_dsl_engine.cpp_stream import compile_formula


def _compiler_version() -> str:
    compiler = os.environ.get("CXX", "g++")
    result = subprocess.run(
        [compiler, "--version"], capture_output=True, text=True, check=False
    )
    text = result.stdout or result.stderr
    return text.splitlines()[0] if text else compiler


def _formulas():
    x = var("x")

    # This intentionally leaves each existing rolling operator untouched.  The
    # windows share the same input and several semantics, so it is a useful test
    # of whether the native optimizer can remove/fuse repeated per-row work by
    # itself.  The temporal mean keeps benchmark output tiny so I/O does not
    # dominate the hot-loop timing.
    rolling = (
        rolling_mean(x, periods=32, min_periods=1)
        + rolling_mean(x, periods=64, min_periods=1)
        + rolling_std(x, periods=32, min_periods=2, ddof=0)
        + rolling_sum(x, periods=32, min_periods=1)
    )
    rolling_shared_input = mean(rolling, axis=0)

    # A second formula is closer to a GP alpha evaluation: stateful temporal
    # transforms, cross-sectional rank, a lag, PnL algebra, then final streaming
    # Sharpe.  It uses the same 5M x 9 source to keep machine/data identical.
    signal_a = xs_rank(ewm(x, span=16))
    signal_b = xs_rank(ewm(x * 1.01 + 0.001, span=37))
    held = shift((signal_a + signal_b) * 0.5, 1, 1)
    pnl = held * x
    gp_like = mean(pnl, axis=0) / std(pnl, axis=0)

    return {
        "rolling_shared_input": rolling_shared_input,
        "gp_like": gp_like,
    }


def _bench_case(name: str, formula, data, output_dir: Path, runs: int, warmups: int):
    started = time.perf_counter()
    runtime = compile_formula(
        formula,
        data,
        n_instruments=data["x"].shape[1],
        prefetch_rows=16,
    )
    compile_seconds = time.perf_counter() - started

    output = output_dir / f"{name}.npy"
    for _ in range(warmups):
        runtime.run(out_path=output, threads=1)

    native_seconds: list[float] = []
    wall_seconds: list[float] = []
    for _ in range(runs):
        wall_started = time.perf_counter()
        result = runtime.run(out_path=output, threads=1)
        wall_seconds.append(time.perf_counter() - wall_started)
        native_seconds.append(float(result.seconds))

    values = np.asarray(np.load(output, mmap_mode="r"))
    generated = runtime.generated_cpp.read_bytes()
    return {
        "case": name,
        "compile_seconds": compile_seconds,
        "native_seconds": native_seconds,
        "median_native_seconds": statistics.median(native_seconds),
        "mean_native_seconds": statistics.mean(native_seconds),
        "min_native_seconds": min(native_seconds),
        "wall_seconds": wall_seconds,
        "median_wall_seconds": statistics.median(wall_seconds),
        "output": values.tolist(),
        "output_shape": list(values.shape),
        "checksum": float(np.nansum(values)),
        "generated_cpp_sha256": hashlib.sha256(generated).hexdigest(),
        "generated_cpp_bytes": len(generated),
        "library_bytes": runtime.library_path.stat().st_size,
        "stages": [stage.kind for stage in runtime.plan.stages],
        "scratch_slots": runtime.plan.scratch_slots,
        "matrix_scratch_slots": runtime.plan.matrix_scratch_slots,
        "parallel_mode": runtime.parallel_plan.mode,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=1)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    x = np.load(args.input, mmap_mode="r")
    data = {"x": x}

    payload = {
        "name": args.name,
        "compiler": os.environ.get("CXX", "g++"),
        "compiler_version": _compiler_version(),
        "compile_flags": os.environ.get("TRADING_DSL_ENGINE_CPP_EXTRA_FLAGS", ""),
        "link_flags": os.environ.get("TRADING_DSL_ENGINE_CPP_EXTRA_LINK_FLAGS", ""),
        "lto": os.environ.get("TRADING_DSL_ENGINE_CPP_LTO", "1"),
        "native": os.environ.get("TRADING_DSL_ENGINE_CPP_NATIVE", "1"),
        "rows": int(x.shape[0]),
        "lanes": int(x.shape[1]),
        "runs": args.runs,
        "warmups": args.warmups,
        "cases": [],
    }
    for case_name, formula in _formulas().items():
        print(f"BENCH_START {args.name} {case_name}", flush=True)
        result = _bench_case(
            case_name, formula, data, args.output_dir, args.runs, args.warmups
        )
        payload["cases"].append(result)
        print(json.dumps(result, sort_keys=True), flush=True)

    path = args.output_dir / "result.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"RESULT_JSON={path}", flush=True)


if __name__ == "__main__":
    main()
