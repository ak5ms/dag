"""Benchmark independent random-GP fitness regions on cpp_stream.

The benchmark compiles bounded multi-output batches of randomly generated,
strongly typed GP alphas.  Every compiled batch is a final-reduction DAG and is
therefore intentionally single-threaded internally; parallelism comes from the
independent batch regions.  It compares:

* serial execution;
* the previous Python ``ThreadPoolExecutor`` orchestration; and
* ``run_many``, the native C++ scheduler used by ``run_gp_alpha_search.py``.

Compilation is outside measured execution.  Every mode is checked against the
serial scores before timings are reported.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
from statistics import median
import tempfile
import time

import numpy as np

from flows.gp import GPConfig, GrammarPolicy, make_pset, random_formula
from flows.riskminer.semantics import gp_alpha_search_terminal_metadata
from trading_dsl_engine.base.dsl import ffill, purify, shift, var, where
from trading_dsl_engine.cpp_stream import compile_formula, run_many


ROWS = int(os.environ.get("GP_SCHEDULER_BENCH_ROWS", "250000"))
N_INSTRUMENTS = int(os.environ.get("GP_SCHEDULER_BENCH_INSTRUMENTS", "9"))
CANDIDATES = int(os.environ.get("GP_SCHEDULER_BENCH_CANDIDATES", "16"))
BATCH_SIZE = int(os.environ.get("GP_SCHEDULER_BENCH_BATCH_SIZE", "2"))
MIN_DEPTH = int(os.environ.get("GP_SCHEDULER_BENCH_MIN_DEPTH", "1"))
MAX_DEPTH = int(os.environ.get("GP_SCHEDULER_BENCH_MAX_DEPTH", "3"))
SEED = int(os.environ.get("GP_SCHEDULER_BENCH_SEED", "1000"))
RUNS = int(os.environ.get("GP_SCHEDULER_BENCH_RUNS", "5"))
WARMUPS = int(os.environ.get("GP_SCHEDULER_BENCH_WARMUPS", "1"))
WORKERS = int(os.environ.get("GP_SCHEDULER_BENCH_WORKERS", "0"))
PIN_WORKERS = os.environ.get("GP_SCHEDULER_BENCH_PIN", "0") == "1"
JSON_PATH = os.environ.get("GP_SCHEDULER_BENCH_JSON", "")


@dataclass(frozen=True, slots=True)
class Sample:
    mode: str
    wall_seconds: float
    cpu_seconds: float
    busy_cores: float
    scheduler_workers: int


@dataclass(frozen=True, slots=True)
class Summary:
    rows: int
    instruments: int
    candidates: int
    batches: int
    batch_size: int
    available_cpus: int
    requested_workers: int
    compile_seconds: float
    warm_compile_seconds: float
    serial_median_seconds: float
    python_pool_median_seconds: float
    native_pool_median_seconds: float
    native_vs_serial_speedup: float
    native_vs_python_speedup: float
    native_median_busy_cores: float
    python_median_busy_cores: float
    formulas: tuple[str, ...]
    plans: tuple[str, ...]
    samples: tuple[Sample, ...]


def _available_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def _sources(rows: int, instruments: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(42)
    returns = rng.normal(0.0, 4.0e-4, size=(rows, instruments))
    returns[::997, 0] = np.nan
    returns[::1237, -1] = 0.0
    price = 100.0 * np.exp(np.cumsum(np.nan_to_num(returns), axis=0))
    positive = rng.lognormal(5.0, 0.45, size=(rows, instruments))
    bounded = rng.uniform(0.05, 0.95, size=(rows, instruments))
    volatility = np.full((rows, instruments), 0.0125, dtype=np.float64)
    tradable = np.broadcast_to(
        ((np.arange(rows) % 1440) < 1380)[:, None],
        (rows, instruments),
    ).astype(np.float64)

    result: dict[str, np.ndarray] = {}
    for name in gp_alpha_search_terminal_metadata():
        lowered = name.lower()
        if "roll_ret" in lowered or lowered.endswith("returns"):
            value = returns
        elif "volume" in lowered or "spread" in lowered:
            value = positive
        elif "pct" in lowered or "ratio" in lowered:
            value = bounded
        else:
            value = price
        result[name] = value
    result.update(
        {
            "roll_rets": returns,
            "clean_rets": returns,
            "volatility": volatility,
            "is_tradable_out0": tradable,
        }
    )
    return result


def _normalized_alpha(alpha):
    return purify(alpha / abs(alpha).sum(axis=-1))


def _fitness_formula(alpha):
    position = shift(_normalized_alpha(alpha) / var("volatility"), 1)
    held = ffill(
        where(
            var("is_tradable_out0"),
            position,
            float("nan"),
        )
    )
    pnl = (shift(held, 1) * var("clean_rets")).sum(axis=1)
    return pnl.mean(axis=0) / pnl.std(axis=0)


def _random_formulas(count: int):
    config = GPConfig(
        grammar=GrammarPolicy(exclude_sections=("utils.group",)),
        fields=gp_alpha_search_terminal_metadata(),
        tensor_fields=(),
    )
    pset = make_pset(config)
    formulas = []
    trees = []
    next_seed = SEED
    failures = []
    while len(formulas) < count and next_seed < SEED + count * 20:
        tree, alpha = random_formula(
            pset,
            min_depth=MIN_DEPTH,
            max_depth=MAX_DEPTH,
            seed=next_seed,
        )
        next_seed += 1
        try:
            formulas.append(_fitness_formula(alpha))
            trees.append(str(tree))
        except Exception as exc:
            failures.append(f"{tree}: {type(exc).__name__}: {exc}")
    if len(formulas) != count:
        raise RuntimeError(
            f"generated {len(formulas)} of {count} formulas; failures={failures[:5]}"
        )
    return formulas, tuple(trees)


def _compile_batches(formulas, sources):
    batches = [
        formulas[start : start + BATCH_SIZE]
        for start in range(0, len(formulas), BATCH_SIZE)
    ]
    started = time.perf_counter()
    runtimes = tuple(
        compile_formula(
            batch,
            sources,
            n_instruments=N_INSTRUMENTS,
            prefetch_rows=16,
        )
        for batch in batches
    )
    compile_seconds = time.perf_counter() - started

    started = time.perf_counter()
    warm = tuple(
        compile_formula(
            batch,
            sources,
            n_instruments=N_INSTRUMENTS,
            prefetch_rows=16,
        )
        for batch in batches
    )
    warm_compile_seconds = time.perf_counter() - started
    if tuple(runtime.library_path for runtime in runtimes) != tuple(
        runtime.library_path for runtime in warm
    ):
        raise RuntimeError("identical GP batches did not reuse native cache")
    return runtimes, compile_seconds, warm_compile_seconds


def _flatten_result(result) -> np.ndarray:
    loaded = result.load(mmap_mode=None)
    values = loaded if isinstance(loaded, tuple) else (loaded,)
    return np.asarray(
        [float(np.asarray(value).reshape(())) for value in values],
        dtype=np.float64,
    )


def _paths(root: Path, mode: str, run: int, count: int) -> list[Path]:
    return [root / f"{mode}_{run:02d}_{index:03d}.npy" for index in range(count)]


def _cleanup(paths) -> None:
    for path in paths:
        path.unlink(missing_ok=True)


def _serial(runtimes, paths) -> tuple[np.ndarray, Sample]:
    started_cpu = time.process_time()
    started = time.perf_counter()
    results = tuple(
        runtime.run(out_path=path, threads=1)
        for runtime, path in zip(runtimes, paths)
    )
    wall = time.perf_counter() - started
    cpu = time.process_time() - started_cpu
    values = np.concatenate([_flatten_result(result) for result in results])
    return values, Sample("serial", wall, cpu, cpu / wall, 1)


def _python_pool(runtimes, paths, workers: int) -> tuple[np.ndarray, Sample]:
    started_cpu = time.process_time()
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(runtime.run, out_path=path, threads=1)
            for runtime, path in zip(runtimes, paths)
        ]
        results = tuple(future.result() for future in futures)
    wall = time.perf_counter() - started
    cpu = time.process_time() - started_cpu
    values = np.concatenate([_flatten_result(result) for result in results])
    return values, Sample("python_pool", wall, cpu, cpu / wall, workers)


def _native_pool(runtimes, paths, workers: int) -> tuple[np.ndarray, Sample]:
    batch = run_many(
        runtimes,
        out_paths=paths,
        workers=workers,
        threads_per_runtime=1,
        pin_workers=PIN_WORKERS,
    )
    values = np.concatenate(
        [_flatten_result(result) for result in batch.results]
    )
    cpu = float(sum(result.cpu_seconds for result in batch.results))
    return values, Sample(
        "native_pool",
        batch.wall_seconds,
        cpu,
        cpu / batch.wall_seconds if batch.wall_seconds else 0.0,
        batch.workers,
    )


def _assert_scores(actual: np.ndarray, expected: np.ndarray) -> None:
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-11,
        atol=1e-12,
        equal_nan=True,
    )


def main() -> None:
    if min(ROWS, N_INSTRUMENTS, CANDIDATES, BATCH_SIZE, RUNS) <= 0:
        raise ValueError("benchmark dimensions and run count must be positive")
    if WARMUPS < 0 or not 0 <= MIN_DEPTH <= MAX_DEPTH:
        raise ValueError("invalid warmup/depth configuration")

    available = _available_cpus()
    workers = max(1, min(WORKERS or available, available))
    sources = _sources(ROWS, N_INSTRUMENTS)
    formulas, trees = _random_formulas(CANDIDATES)
    runtimes, compile_seconds, warm_compile_seconds = _compile_batches(
        formulas,
        sources,
    )
    plans = tuple(
        f"{runtime.parallel_plan.mode}: {runtime.parallel_plan.reason}"
        for runtime in runtimes
    )

    samples: list[Sample] = []
    with tempfile.TemporaryDirectory(prefix="gp-native-scheduler-") as directory:
        root = Path(directory)
        reference = None
        for warmup in range(WARMUPS):
            paths = _paths(root, "warmup", warmup, len(runtimes))
            reference, _ = _serial(runtimes, paths)
            _cleanup(paths)

        for run in range(RUNS):
            order = ("serial", "python_pool", "native_pool")
            order = order[run % len(order) :] + order[: run % len(order)]
            for mode in order:
                paths = _paths(root, mode, run, len(runtimes))
                if mode == "serial":
                    values, sample = _serial(runtimes, paths)
                    if reference is None:
                        reference = values
                elif mode == "python_pool":
                    values, sample = _python_pool(runtimes, paths, workers)
                else:
                    values, sample = _native_pool(runtimes, paths, workers)
                assert reference is not None
                _assert_scores(values, reference)
                samples.append(sample)
                _cleanup(paths)
                print(json.dumps(asdict(sample), sort_keys=True), flush=True)

    by_mode = {
        mode: [sample for sample in samples if sample.mode == mode]
        for mode in ("serial", "python_pool", "native_pool")
    }
    medians = {
        mode: median(sample.wall_seconds for sample in values)
        for mode, values in by_mode.items()
    }
    native_busy = median(
        sample.busy_cores for sample in by_mode["native_pool"]
    )
    python_busy = median(
        sample.busy_cores for sample in by_mode["python_pool"]
    )
    summary = Summary(
        rows=ROWS,
        instruments=N_INSTRUMENTS,
        candidates=CANDIDATES,
        batches=len(runtimes),
        batch_size=BATCH_SIZE,
        available_cpus=available,
        requested_workers=workers,
        compile_seconds=compile_seconds,
        warm_compile_seconds=warm_compile_seconds,
        serial_median_seconds=medians["serial"],
        python_pool_median_seconds=medians["python_pool"],
        native_pool_median_seconds=medians["native_pool"],
        native_vs_serial_speedup=medians["serial"] / medians["native_pool"],
        native_vs_python_speedup=(
            medians["python_pool"] / medians["native_pool"]
        ),
        native_median_busy_cores=native_busy,
        python_median_busy_cores=python_busy,
        formulas=trees,
        plans=plans,
        samples=tuple(samples),
    )
    payload = asdict(summary)
    print(json.dumps({"summary": payload}, indent=2, sort_keys=True))
    if JSON_PATH:
        path = Path(JSON_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    if available > 1 and summary.native_vs_serial_speedup <= 1.20:
        raise RuntimeError(
            "native scheduler failed to materially outperform serial execution"
        )
    if available > 1 and native_busy <= 1.20:
        raise RuntimeError("native scheduler did not use multiple CPU cores")


if __name__ == "__main__":
    main()
