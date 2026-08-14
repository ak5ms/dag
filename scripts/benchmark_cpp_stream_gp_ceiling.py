from __future__ import annotations

import ctypes
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import resource
from statistics import mean, median
import subprocess
import tempfile
import time

import numpy as np

from flows.riskminer.cpp_stream_eval import (
    CppStreamCandidateEvaluator,
    EvaluationSummary,
    build_candidate_score_formula,
)
from trading_dsl_engine.base.dsl import cat, emit, var
from trading_dsl_engine.cpp_stream import compile_formula
from trading_dsl_engine.cpp_stream.python.compile import _cache_root, _compiler, _flags


ROWS = int(os.environ.get("CPP_STREAM_GP_CEILING_ROWS", "5000000"))
N = int(os.environ.get("CPP_STREAM_GP_CEILING_INSTRUMENTS", "9"))
RUNS = int(os.environ.get("CPP_STREAM_GP_CEILING_RUNS", "10"))
WARMUPS = int(os.environ.get("CPP_STREAM_GP_CEILING_WARMUPS", "1"))
THREADS = int(os.environ.get("CPP_STREAM_GP_CEILING_THREADS", "0"))
OUTPUT_DIR = Path(os.environ.get("CPP_STREAM_BENCH_OUTPUT_DIR", "/dev/shm"))
JSON_PATH = os.environ.get("CPP_STREAM_GP_CEILING_JSON", "")
POPULATION_ROWS = int(os.environ.get("CPP_STREAM_GP_POPULATION_ROWS", "500000"))
POPULATION_CANDIDATES = int(os.environ.get("CPP_STREAM_GP_POPULATION_CANDIDATES", "16"))
POPULATION_BATCH_SIZE = int(os.environ.get("CPP_STREAM_GP_POPULATION_BATCH_SIZE", "4"))
POPULATION_WORKERS = int(os.environ.get("CPP_STREAM_GP_POPULATION_WORKERS", "4"))
SOURCE = Path(__file__).with_name("cpp_stream_reduction_native_ceiling.cpp")


@dataclass(frozen=True)
class Case:
    name: str
    formula: object
    native_name: str
    output_size: int
    expected_mode: str


def _rss_mb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _create_matrix(path: Path, seed: int, scale: float = 1.0) -> Path:
    rng = np.random.default_rng(seed)
    values = np.lib.format.open_memmap(path, mode="w+", dtype=np.float64, shape=(ROWS, N))
    for begin in range(0, ROWS, 65_536):
        end = min(begin + 65_536, ROWS)
        values[begin:end] = rng.normal(scale=scale, size=(end - begin, N))
    values.flush()
    del values
    return path


def _native_library(root: Path) -> tuple[ctypes.CDLL, dict[str, object]]:
    compiler = _compiler()
    compile_flags, link_flags = _flags()
    ceiling_define = f"-DCPP_STREAM_CEILING_N={N}"
    digest = hashlib.sha256(SOURCE.read_bytes())
    digest.update("\0".join((*compile_flags, ceiling_define, *link_flags)).encode())
    library = root / f"reduction-ceiling-{digest.hexdigest()[:16]}.so"
    started = time.perf_counter()
    subprocess.run([compiler, *compile_flags, ceiling_define, str(SOURCE), *link_flags, "-o", str(library)], check=True)
    compile_seconds = time.perf_counter() - started
    loaded = ctypes.CDLL(str(library))
    pointer = ctypes.POINTER(ctypes.c_double)
    loaded.column_stats_ceiling.argtypes = [pointer, ctypes.c_size_t, ctypes.c_size_t, pointer]
    for name in ("stateless_sharpe_ceiling", "shifted_alpha_sharpe_ceiling"):
        getattr(loaded, name).argtypes = [pointer, pointer, ctypes.c_size_t, ctypes.c_size_t, pointer]
    return loaded, {
        "compiler": compiler,
        "compile_flags": [*compile_flags, ceiling_define],
        "link_flags": link_flags,
        "compile_seconds": compile_seconds,
        "library": str(library),
    }


def _compile_runtime(case: Case, data: dict[str, Path]):
    started = time.perf_counter()
    runtime = compile_formula(case.formula, data, n_instruments=N, prefetch_rows=16)
    cold_seconds = time.perf_counter() - started
    started = time.perf_counter()
    warm_runtime = compile_formula(case.formula, data, n_instruments=N, prefetch_rows=16)
    warm_seconds = time.perf_counter() - started
    if warm_runtime.library_path != runtime.library_path:
        raise RuntimeError("identical formula did not reuse the native cache")
    return runtime, cold_seconds, warm_seconds


def _native_arguments(case: Case, library: ctypes.CDLL, x: np.ndarray, y: np.ndarray, output: np.ndarray):
    pointer = ctypes.POINTER(ctypes.c_double)
    function = getattr(library, case.native_name)
    if case.native_name == "column_stats_ceiling":
        args = (x.ctypes.data_as(pointer), ROWS, N, output.ctypes.data_as(pointer))
    else:
        args = (x.ctypes.data_as(pointer), y.ctypes.data_as(pointer), ROWS, N, output.ctypes.data_as(pointer))
    return function, args


def _measure_native(function, args) -> list[float]:
    for _ in range(WARMUPS):
        function(*args)
    timings = []
    for _ in range(RUNS):
        started = time.perf_counter()
        function(*args)
        timings.append(time.perf_counter() - started)
    return timings


def _measure_runtime(runtime, path: Path, threads: int) -> list[object]:
    for _ in range(WARMUPS):
        runtime.run(out_path=path, threads=threads, pin_threads=True, async_writeback_mb=0)
    return [
        runtime.run(out_path=path, threads=threads, pin_threads=True, async_writeback_mb=0)
        for _ in range(RUNS)
    ]


def _result_values(result) -> np.ndarray:
    return np.fromfile(result.output_path, dtype=np.float64).reshape(-1)


def _case_summary(case: Case, runtime, serial_runs, automatic_runs, native_times,
                  native_output, cold_compile: float, warm_compile: float) -> dict[str, object]:
    serial_seconds = median(item.seconds for item in serial_runs)
    automatic_seconds = median(item.seconds for item in automatic_runs)
    native_seconds = median(native_times)
    serial_output = _result_values(serial_runs[-1])
    automatic_output = _result_values(automatic_runs[-1])
    np.testing.assert_allclose(serial_output, native_output.reshape(-1), rtol=8e-12, atol=8e-12, equal_nan=True)
    np.testing.assert_allclose(automatic_output, serial_output, rtol=8e-12, atol=8e-12, equal_nan=True)
    if runtime.parallel_plan.mode != case.expected_mode:
        raise RuntimeError(
            f"{case.name} planned as {runtime.parallel_plan.mode!r}, expected {case.expected_mode!r}: "
            f"{runtime.parallel_plan.reason}"
        )
    return {
        "case": case.name,
        "rows": ROWS,
        "instruments": N,
        "output_size": case.output_size,
        "parallel_mode": runtime.parallel_plan.mode,
        "parallel_reason": runtime.parallel_plan.reason,
        "work_score": runtime.parallel_plan.work_score,
        "cold_compile_seconds": cold_compile,
        "warm_compile_seconds": warm_compile,
        "generated_cpp": str(runtime.generated_cpp),
        "library_path": str(runtime.library_path),
        "serial_threads": serial_runs[-1].threads,
        "automatic_threads": automatic_runs[-1].threads,
        "serial_median_seconds": serial_seconds,
        "automatic_median_seconds": automatic_seconds,
        "native_ceiling_median_seconds": native_seconds,
        "serial_million_rows_per_second": ROWS / serial_seconds / 1e6,
        "automatic_million_rows_per_second": ROWS / automatic_seconds / 1e6,
        "native_ceiling_million_rows_per_second": ROWS / native_seconds / 1e6,
        "automatic_speedup": serial_seconds / automatic_seconds,
        "serial_fraction_of_single_thread_ceiling": native_seconds / serial_seconds,
        "automatic_fraction_of_single_thread_ceiling": native_seconds / automatic_seconds,
        "serial_runs_seconds": [item.seconds for item in serial_runs],
        "automatic_runs_seconds": [item.seconds for item in automatic_runs],
        "native_runs_seconds": native_times,
        "serial_busy_cores": [item.average_busy_cores for item in serial_runs],
        "automatic_busy_cores": [item.average_busy_cores for item in automatic_runs],
        "checksum": float(np.nansum(serial_output)),
        "rss_mb": _rss_mb(),
    }


def _population_candidates(count: int):
    x = var("x")
    y = var("y")
    result = []
    for index in range(count):
        a = 0.05 * (index + 1)
        b = 0.03 * ((index % 7) + 1)
        selector = index % 6
        if selector == 0:
            expression = x + a
        elif selector == 1:
            expression = y - a
        elif selector == 2:
            expression = x + y * a
        elif selector == 3:
            expression = x - y * a
        elif selector == 4:
            expression = (x + a) * (y - b)
        else:
            expression = (x - y) * (x + a) + b
        result.append(expression)
    return tuple(result)


def _reset_evaluator(evaluator: CppStreamCandidateEvaluator) -> None:
    evaluator.score_cache.clear()
    evaluator.summary = EvaluationSummary()


def _run_population_once(evaluator: CppStreamCandidateEvaluator, candidates):
    _reset_evaluator(evaluator)
    started = time.perf_counter()
    scores = evaluator.evaluate(candidates)
    wall_seconds = time.perf_counter() - started
    summary = evaluator.summary
    return wall_seconds, np.asarray(tuple(scores.values()), dtype=np.float64), {
        "compile_seconds": summary.compile_seconds,
        "sum_native_seconds": sum(float(batch.native_seconds or 0.0) for batch in summary.batches),
        "batch_count": len(summary.batches),
        "peak_batch_workers": summary.peak_batch_workers,
        "requested_native_threads": sorted({batch.requested_threads for batch in summary.batches}),
        "actual_native_threads": sorted({int(batch.actual_threads) for batch in summary.batches if batch.actual_threads is not None}),
        "finite": summary.finite,
        "nonfinite": summary.nonfinite,
        "compile_rejected": summary.compile_rejected,
        "execution_rejected": summary.execution_rejected,
    }


def _benchmark_population(root: Path, x: np.ndarray, y: np.ndarray) -> dict[str, object]:
    rows = min(ROWS, POPULATION_ROWS)
    data = {"x": np.asarray(x[:rows]), "y": np.asarray(y[:rows])}
    candidates = _population_candidates(POPULATION_CANDIDATES)
    serial = CppStreamCandidateEvaluator(
        data, n_instruments=N, work_dir=root / "population-serial", roll_rets_name="y",
        batch_size=POPULATION_BATCH_SIZE, workers=1, compile_kwargs={"prefetch_rows": 16},
    )
    parallel = CppStreamCandidateEvaluator(
        data, n_instruments=N, work_dir=root / "population-parallel", roll_rets_name="y",
        batch_size=POPULATION_BATCH_SIZE, workers=POPULATION_WORKERS, compile_kwargs={"prefetch_rows": 16},
    )
    warmups = {"serial": [], "parallel": []}
    for evaluator, name in ((serial, "serial"), (parallel, "parallel")):
        for _ in range(WARMUPS):
            wall, values, details = _run_population_once(evaluator, candidates)
            warmups[name].append({**details, "wall_seconds": wall})
            if values.size != POPULATION_CANDIDATES:
                raise RuntimeError("candidate population emitted the wrong size")
    timings = {"serial": [], "parallel": []}
    details = {"serial": [], "parallel": []}
    reference = None
    for repetition in range(RUNS):
        order = ((serial, "serial"), (parallel, "parallel")) if repetition % 2 == 0 else ((parallel, "parallel"), (serial, "serial"))
        outputs = {}
        for evaluator, name in order:
            wall, values, record = _run_population_once(evaluator, candidates)
            timings[name].append(wall)
            details[name].append(record)
            outputs[name] = values
        np.testing.assert_allclose(outputs["parallel"], outputs["serial"], rtol=8e-12, atol=8e-12, equal_nan=True)
        if reference is None:
            reference = outputs["serial"].copy()
        else:
            np.testing.assert_allclose(outputs["serial"], reference, rtol=0.0, atol=0.0, equal_nan=True)
    serial_median = median(timings["serial"])
    parallel_median = median(timings["parallel"])
    parallel_peaks = [int(item["peak_batch_workers"]) for item in details["parallel"]]
    if POPULATION_WORKERS > 1 and len(candidates) > POPULATION_BATCH_SIZE and max(parallel_peaks, default=1) < 2:
        raise RuntimeError("candidate population did not use parallel batches")
    return {
        "case": "gp_candidate_population",
        "rows": rows,
        "instruments": N,
        "candidate_count": len(candidates),
        "batch_size": POPULATION_BATCH_SIZE,
        "requested_population_workers": POPULATION_WORKERS,
        "serial_median_seconds": serial_median,
        "parallel_median_seconds": parallel_median,
        "speedup": serial_median / parallel_median,
        "serial_candidates_per_second": len(candidates) / serial_median,
        "parallel_candidates_per_second": len(candidates) / parallel_median,
        "serial_runs_seconds": timings["serial"],
        "parallel_runs_seconds": timings["parallel"],
        "serial_details": details["serial"],
        "parallel_details": details["parallel"],
        "warmups": warmups,
        "checksum": float(np.nansum(reference)) if reference is not None else 0.0,
        "rss_mb": _rss_mb(),
    }


def main() -> None:
    if min(ROWS, N, RUNS) <= 0 or WARMUPS < 0 or THREADS < 0:
        raise ValueError("rows, instruments, runs must be positive; warmups/threads nonnegative")
    if min(POPULATION_ROWS, POPULATION_CANDIDATES, POPULATION_BATCH_SIZE) <= 0 or POPULATION_WORKERS < 1:
        raise ValueError("invalid population benchmark dimensions")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metadata = {
        "github_sha": os.environ.get("GITHUB_SHA"),
        "github_head_ref": os.environ.get("GITHUB_HEAD_REF"),
        "github_base_ref": os.environ.get("GITHUB_BASE_REF"),
        "rows": ROWS,
        "instruments": N,
        "warmups": WARMUPS,
        "runs": RUNS,
        "automatic_threads_request": THREADS,
        "cache_root": str(_cache_root()),
        "population_rows": POPULATION_ROWS,
        "population_candidates": POPULATION_CANDIDATES,
        "population_batch_size": POPULATION_BATCH_SIZE,
        "population_workers": POPULATION_WORKERS,
    }
    with tempfile.TemporaryDirectory(prefix="cpp-stream-gp-ceiling-", dir=OUTPUT_DIR) as temporary:
        root = Path(temporary)
        x_path = _create_matrix(root / "x.npy", seed=20260813)
        y_path = _create_matrix(root / "y.npy", seed=20260814, scale=0.01)
        data = {"x": x_path, "y": y_path}
        x = np.load(x_path, mmap_mode="r")
        y = np.load(y_path, mmap_mode="r")
        library, native_metadata = _native_library(root)
        metadata["native"] = native_metadata
        x_expr = var("x")
        y_expr = var("y")
        pnl = (x_expr * y_expr).sum(axis=1)
        cases = (
            Case("column_sum_mean_std", cat(x_expr.sum(axis=0), x_expr.mean(axis=0), x_expr.std(axis=0, ddof=0)), "column_stats_ceiling", 3 * N, "rows"),
            Case("stateless_alpha_sharpe", emit(pnl.mean(axis=0) / pnl.std(axis=0, ddof=0), mode="last"), "stateless_sharpe_ceiling", 1, "rows"),
            Case("shifted_candidate_sharpe", build_candidate_score_formula([x_expr], roll_rets_name="y"), "shifted_alpha_sharpe_ceiling", 1, "serial"),
        )
        results = []
        for case in cases:
            runtime, cold_compile, warm_compile = _compile_runtime(case, data)
            native_output = np.empty(case.output_size, dtype=np.float64)
            function, arguments = _native_arguments(case, library, x, y, native_output)
            native_times = _measure_native(function, arguments)
            serial_runs = _measure_runtime(runtime, root / f"{case.name}-serial.bin", 1)
            automatic_runs = _measure_runtime(runtime, root / f"{case.name}-automatic.bin", THREADS)
            result = _case_summary(case, runtime, serial_runs, automatic_runs, native_times, native_output, cold_compile, warm_compile)
            results.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)
        population = _benchmark_population(root, x, y)
        print(json.dumps(population, sort_keys=True), flush=True)
    report = {"metadata": metadata, "results": results, "population": population}
    print(json.dumps({"summary": {
        "cases": len(results),
        "mean_serial_fraction_of_ceiling": mean(float(item["serial_fraction_of_single_thread_ceiling"]) for item in results),
        "mean_automatic_speedup": mean(float(item["automatic_speedup"]) for item in results),
        "max_rss_mb": max([float(item["rss_mb"]) for item in results] + [float(population["rss_mb"])]),
        "population_speedup": population["speedup"],
    }}, sort_keys=True), flush=True)
    if JSON_PATH:
        path = Path(JSON_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
