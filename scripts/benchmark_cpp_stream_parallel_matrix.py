from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
import tempfile

import numpy as np

from flows.riskmodel import roll_rets
from trading_dsl_engine.base.dsl import (
    Ridge,
    cat,
    cumsum,
    ewm,
    einsum,
    get_beta,
    groupby,
    self_,
    univ,
    var,
    xs_rank,
)
from trading_dsl_engine.base.keys import Key
from trading_dsl_engine.cpp_stream import compile_formula


ROWS = int(os.environ.get("CPP_STREAM_PARALLEL_ROWS", "1000000"))
ROLL_ROWS = int(os.environ.get("CPP_STREAM_PARALLEL_ROLL_ROWS", str(ROWS)))
N = int(os.environ.get("CPP_STREAM_PARALLEL_INSTRUMENTS", "64"))
RUNS = int(os.environ.get("CPP_STREAM_PARALLEL_RUNS", "5"))
WARMUPS = int(os.environ.get("CPP_STREAM_PARALLEL_WARMUPS", "1"))
PREFETCH_ROWS = int(os.environ.get("CPP_STREAM_PARALLEL_PREFETCH_ROWS", "16"))
OUTPUT_DIR = Path(os.environ.get("CPP_STREAM_BENCH_OUTPUT_DIR", "/dev/shm"))
CASE_TEXT = os.environ.get("CPP_STREAM_PARALLEL_CASE", "all")
THREAD_TEXT = os.environ.get("CPP_STREAM_PARALLEL_THREADS", "1,2,4")

CASE_NAMES = (
    "deep_elementwise",
    "optimized_einsum",
    "stateless_ridge",
    "stateful_ewm",
    "grouped_state",
    "temporal_rank_serial",
    "roll_rets",
)


@dataclass(frozen=True)
class Workload:
    formula: object
    data: dict[str, Path]
    rows: int
    instruments: int
    group_capacity: int = 64


def available_cpus() -> tuple[int, tuple[int, ...]]:
    try:
        ids = tuple(sorted(os.sched_getaffinity(0)))
    except AttributeError:
        ids = tuple(range(os.cpu_count() or 1))
    return len(ids), ids


def selected_cases() -> tuple[str, ...]:
    if CASE_TEXT == "all":
        return CASE_NAMES
    selected = tuple(value.strip() for value in CASE_TEXT.split(",") if value.strip())
    unknown = sorted(set(selected) - set(CASE_NAMES))
    if not selected or unknown:
        raise ValueError(f"invalid CPP_STREAM_PARALLEL_CASE={CASE_TEXT!r}; unknown={unknown}")
    return selected


def thread_counts(available: int) -> tuple[int, ...]:
    requested = [int(value.strip()) for value in THREAD_TEXT.split(",") if value.strip()]
    return tuple(sorted({1, *(max(1, min(value, available)) for value in requested)}))


def create_matrix(path: Path, rows: int, n: int, seed: int) -> Path:
    rng = np.random.default_rng(seed)
    array = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float64, shape=(rows, n)
    )
    for start in range(0, rows, 65_536):
        stop = min(start + 65_536, rows)
        array[start:stop] = rng.normal(size=(stop - start, n))
    array.flush()
    del array
    return path


def base_inputs(root: Path) -> dict[str, Path]:
    return {
        name: create_matrix(root / f"{name}.npy", ROWS, N, seed)
        for name, seed in (("w", 1), ("x", 2), ("y", 3), ("z", 4))
    }


def minute_input(root: Path) -> Path:
    path = root / "minute_key.npy"
    values = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.int64, shape=(ROWS,)
    )
    values[:] = np.remainder(np.arange(ROWS, dtype=np.int64), 60)
    values.flush()
    del values
    return path


def roll_inputs(root: Path) -> dict[str, Path]:
    n = 9
    names = (
        "_ev_ts",
        "session_start0",
        "session_end0",
        "volume_out0",
        "is_tradable_out0",
        "is_tradable_out1",
        "wdte_out0",
        "mp_out0.close",
        "mp_out1.close",
    )
    paths = {name: root / f"roll_{index}.npy" for index, name in enumerate(names)}
    scalar = {"_ev_ts", "session_start0", "session_end0", "wdte_out0"}
    arrays = {
        name: np.lib.format.open_memmap(
            paths[name],
            mode="w+",
            dtype=np.float64,
            shape=(ROLL_ROWS,) if name in scalar else (ROLL_ROWS, n),
        )
        for name in names
    }
    minute_us = 60_000_000.0
    day_us = 86_400_000_000.0
    session_minutes = 1440
    base = 1_700_000_000_000_000.0
    lane = np.arange(n, dtype=np.float64)[None, :]
    for start in range(0, ROLL_ROWS, 65_536):
        stop = min(start + 65_536, ROLL_ROWS)
        t = np.arange(start, stop, dtype=np.float64)
        day = np.floor_divide(t.astype(np.int64), session_minutes)
        minute = np.remainder(t.astype(np.int64), session_minutes)
        session_start = base + day.astype(np.float64) * day_us
        event_ts = session_start + minute.astype(np.float64) * minute_us
        session_end = session_start + day_us
        weekday = (np.remainder(day + 2, 7) < 5).astype(np.float64)
        tradable_scalar = ((minute >= 60) & (minute < 1380)).astype(np.float64) * weekday
        phase = minute.astype(np.float64)[:, None] / session_minutes
        tradable = tradable_scalar[:, None] * np.ones((1, n))
        volume = np.maximum(
            100.0 + 25.0 * np.sin(2.0 * np.pi * phase) + lane,
            0.0,
        ) * tradable
        time_column = t[:, None]
        arrays["_ev_ts"][start:stop] = event_ts
        arrays["session_start0"][start:stop] = session_start
        arrays["session_end0"][start:stop] = session_end
        arrays["volume_out0"][start:stop] = volume
        arrays["is_tradable_out0"][start:stop] = tradable
        arrays["is_tradable_out1"][start:stop] = tradable
        arrays["wdte_out0"][start:stop] = np.where(np.remainder(day, 5) == 0, 1.0, 2.0)
        arrays["mp_out0.close"][start:stop] = 100.0 + 0.0010 * time_column + 0.01 * lane
        arrays["mp_out1.close"][start:stop] = 101.0 + 0.0011 * time_column + 0.01 * lane
    for array in arrays.values():
        array.flush()
    arrays.clear()
    return paths


def build_workload(
    name: str,
    root: Path,
    base: dict[str, Path] | None,
) -> Workload:
    if name == "roll_rets":
        return Workload(roll_rets, roll_inputs(root), ROLL_ROWS, 9, 4096)
    if base is None:
        raise AssertionError("base inputs were not created")
    w, x, y, z = (var(field) for field in ("w", "x", "y", "z"))
    if name == "deep_elementwise":
        formula = ((x * 1.5 + y) ** 2) / (z * z + 0.25) + w * x - y / 3.0
        return Workload(formula, base, ROWS, N)
    if name == "optimized_einsum":
        formula = einsum(
            "ij,kj,kl->il",
            cat(x, y),
            cat(y, z),
            cat(z, w),
            optimize="optimal",
        )
        return Workload(formula, base, ROWS, N)
    if name == "stateless_ridge":
        formula = get_beta(Ridge(cat(x, y, z), y=w, hl=0, lambda_=0.1))
        return Workload(formula, base, ROWS, N)
    if name == "stateful_ewm":
        return Workload(ewm(x, 21), {"x": base["x"]}, ROWS, N)
    if name == "grouped_state":
        formula = groupby(
            (
                univ(list(range(N))),
                Key(
                    var("minute_key"),
                    num_keys=60,
                    row_scalar=True,
                    dtype="int64",
                ),
            ),
            x,
            ewm(cumsum(self_), 3),
        )
        return Workload(
            formula,
            {"x": base["x"], "minute_key": minute_input(root)},
            ROWS,
            N,
        )
    if name == "temporal_rank_serial":
        return Workload(xs_rank(ewm(x, 21)), {"x": base["x"]}, ROWS, N)
    raise AssertionError(name)


def output_stats(path: Path, rows: int, shape: tuple[int, ...]) -> tuple[float, float]:
    values = np.memmap(path, mode="r", dtype=np.float64, shape=(rows,) + shape)
    tail = values[-min(rows, 8192) :]
    result = float(np.nansum(tail)), float(np.isfinite(tail).mean())
    del values
    return result


def benchmark(
    name: str,
    workload: Workload,
    counts: tuple[int, ...],
) -> list[dict[str, object]]:
    runtime = compile_formula(
        workload.formula,
        workload.data,
        n_instruments=workload.instruments,
        default_group_capacity=workload.group_capacity,
        prefetch_rows=PREFETCH_ROWS,
    )
    results: list[dict[str, object]] = []
    reference_checksum: float | None = None
    for requested in counts:
        output = OUTPUT_DIR / f"parallel_matrix_{name}_{requested}.bin"
        for _ in range(WARMUPS):
            runtime.run(out_path=output, threads=requested, pin_threads=True)
        runs = [
            runtime.run(out_path=output, threads=requested, pin_threads=True)
            for _ in range(RUNS)
        ]
        checksum, finite_fraction = output_stats(
            output, workload.rows, tuple(runtime.plan.output_shape)
        )
        if reference_checksum is None:
            reference_checksum = checksum
        elif not np.isclose(checksum, reference_checksum, rtol=1e-11, atol=1e-11):
            raise RuntimeError(
                f"checksum mismatch for {name}: {checksum} != {reference_checksum}"
            )
        rates = [run.rows_per_second for run in runs]
        busy = [run.average_busy_cores for run in runs]
        results.append(
            {
                "case": name,
                "mode": runtime.parallel_plan.mode,
                "reason": runtime.parallel_plan.reason,
                "requested": requested,
                "actual": runs[0].threads,
                "available": runs[0].available_cpus,
                "median_rate": median(rates),
                "mean_rate": mean(rates),
                "best_rate": max(rates),
                "median_busy": median(busy),
                "median_cpu": median(run.cpu_seconds for run in runs),
                "rates": rates,
                "busy": busy,
                "checksum": checksum,
                "finite_fraction": finite_fraction,
                "shape": runtime.plan.output_shape,
            }
        )
        output.unlink(missing_ok=True)
    baseline = float(results[0]["median_rate"])
    for result in results:
        result["speedup"] = float(result["median_rate"]) / baseline
        result["efficiency"] = float(result["speedup"]) / int(result["actual"])
    return results


def main() -> None:
    available, cpu_ids = available_cpus()
    if available < 2:
        raise SystemExit(f"requires multiple CPUs; affinity={cpu_ids}")
    counts = thread_counts(available)
    if max(counts) < 2:
        raise SystemExit("thread list contains no multicore run")
    cases = selected_cases()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"available_cpus={available} cpu_ids={cpu_ids}")
    print(f"thread_counts={counts} pin_threads=True")
    print(f"rows={ROWS:,} roll_rows={ROLL_ROWS:,} instruments={N}")
    print(f"warmups={WARMUPS} runs={RUNS}")

    with tempfile.TemporaryDirectory(prefix="cpp_stream_parallel_matrix_") as temp:
        root = Path(temp)
        needs_base = any(name != "roll_rets" for name in cases)
        base = base_inputs(root) if needs_base else None
        all_results: list[dict[str, object]] = []
        for name in cases:
            all_results.extend(benchmark(name, build_workload(name, root, base), counts))

    for result in all_results:
        print("---")
        print(f"case={result['case']}")
        print(f"mode={result['mode']} reason={result['reason']}")
        print(
            f"requested_threads={result['requested']} actual_threads={result['actual']} "
            f"available_cpus={result['available']}"
        )
        print(f"output_shape={result['shape']}")
        print(f"median={float(result['median_rate']) / 1e6:.6f} M rows/s")
        print(f"mean={float(result['mean_rate']) / 1e6:.6f} M rows/s")
        print(f"best={float(result['best_rate']) / 1e6:.6f} M rows/s")
        print(f"speedup={float(result['speedup']):.4f}x")
        print(f"parallel_efficiency={100.0 * float(result['efficiency']):.2f}%")
        print(f"median_busy_cores={float(result['median_busy']):.3f}")
        print(f"median_cpu_seconds={float(result['median_cpu']):.6f}")
        print("runs=" + ", ".join(f"{rate / 1e6:.6f}" for rate in result["rates"]) + " M rows/s")
        print("busy_cores=" + ", ".join(f"{value:.3f}" for value in result["busy"]))
        print(f"checksum={float(result['checksum']):.12g}")
        print(f"tail_finite_fraction={float(result['finite_fraction']):.12g}")

    multicore = [
        result
        for result in all_results
        if int(result["actual"]) >= 2 and result["mode"] != "serial"
    ]
    if not multicore:
        raise SystemExit("no workload executed with multiple threads")
    if max(float(result["median_busy"]) for result in multicore) <= 1.1:
        raise SystemExit("no workload demonstrated more than one busy core")


if __name__ == "__main__":
    main()
