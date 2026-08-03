from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
import tempfile

import numpy as np

from trading_dsl_engine.base.dsl import cat, ewm, var
from trading_dsl_engine.cpp_stream import compile_formula


ROWS = int(os.environ.get("CPP_STREAM_PARALLEL_REDUCTION_ROWS", "5000000"))
N = int(os.environ.get("CPP_STREAM_PARALLEL_REDUCTION_INSTRUMENTS", "9"))
RUNS = int(os.environ.get("CPP_STREAM_PARALLEL_REDUCTION_RUNS", "10"))
WARMUPS = int(os.environ.get("CPP_STREAM_PARALLEL_REDUCTION_WARMUPS", "1"))
THREAD_TEXT = os.environ.get("CPP_STREAM_PARALLEL_REDUCTION_THREADS", "1,2,4")
CASE_TEXT = os.environ.get("CPP_STREAM_PARALLEL_REDUCTION_CASE", "all")
OUTPUT_DIR = Path(os.environ.get("CPP_STREAM_BENCH_OUTPUT_DIR", "/dev/shm"))
ROW_MIN_SPEEDUP = float(
    os.environ.get("CPP_STREAM_PARALLEL_REDUCTION_ROW_MIN_SPEEDUP", "1.15")
)
LANE_MIN_SPEEDUP = float(
    os.environ.get("CPP_STREAM_PARALLEL_REDUCTION_LANE_MIN_SPEEDUP", "1.05")
)


@dataclass(frozen=True)
class Workload:
    name: str
    formula: object
    data: dict[str, Path]
    expected_mode: str
    minimum_speedup: float


CASE_NAMES = (
    "row_sum",
    "row_mean",
    "row_std",
    "lane_feature_sum",
    "lane_feature_std",
)


def available_cpu_ids() -> tuple[int, ...]:
    try:
        return tuple(sorted(os.sched_getaffinity(0)))
    except AttributeError:
        return tuple(range(os.cpu_count() or 1))


def thread_counts(available: int) -> tuple[int, ...]:
    requested = [
        int(value.strip())
        for value in THREAD_TEXT.split(",")
        if value.strip()
    ]
    counts = tuple(
        sorted({1, *(max(1, min(value, available)) for value in requested)})
    )
    if counts[-1] < 2:
        raise SystemExit("parallel reduction benchmark requires at least two threads")
    return counts


def selected_cases() -> tuple[str, ...]:
    if CASE_TEXT == "all":
        return CASE_NAMES
    selected = tuple(
        value.strip() for value in CASE_TEXT.split(",") if value.strip()
    )
    unknown = sorted(set(selected) - set(CASE_NAMES))
    if not selected or unknown:
        raise ValueError(
            f"invalid CPP_STREAM_PARALLEL_REDUCTION_CASE={CASE_TEXT!r}; "
            f"unknown={unknown}"
        )
    return selected


def create_matrix(path: Path, seed: int) -> Path:
    rng = np.random.default_rng(seed)
    values = np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=np.float64,
        shape=(ROWS, N),
    )
    for start in range(0, ROWS, 65_536):
        stop = min(start + 65_536, ROWS)
        values[start:stop] = rng.normal(size=(stop - start, N))
    values.flush()
    del values
    return path


def build_workloads(root: Path) -> dict[str, Workload]:
    data = {
        name: create_matrix(root / f"{name}.npy", seed)
        for name, seed in (("x", 1), ("y", 2), ("z", 3))
    }
    x, y, z = (var(name) for name in ("x", "y", "z"))

    # Reducing axis 1 removes the instrument dimension independently for each
    # row. The complete plan is therefore safely row-sharded.
    stateless_features = cat(
        x * 1.01 + y,
        x - y * 0.1,
        x * y,
        (x + y) * 0.5,
        (x - z) ** 2,
        y / (z * z + 0.25),
        x + 5.0,
        y * 3.0 - z,
    )

    # Reducing axis 2 removes only the feature dimension. The EWM state stays
    # instrument-local, so workers can own disjoint lane ranges for all rows.
    temporal_features = cat(
        ewm(x * 1.01 + y, 8),
        ewm(x - y * 0.1, 12),
        ewm(x * y, 16),
        ewm((x + y) * 0.5, 24),
        ewm((x - z) ** 2, 32),
        ewm(y / (z * z + 0.25), 48),
    )

    return {
        "row_sum": Workload(
            "row_sum",
            stateless_features.sum(axis=1),
            data,
            "rows",
            ROW_MIN_SPEEDUP,
        ),
        "row_mean": Workload(
            "row_mean",
            stateless_features.mean(axis=1),
            data,
            "rows",
            ROW_MIN_SPEEDUP,
        ),
        "row_std": Workload(
            "row_std",
            stateless_features.std(axis=1),
            data,
            "rows",
            ROW_MIN_SPEEDUP,
        ),
        "lane_feature_sum": Workload(
            "lane_feature_sum",
            temporal_features.sum(axis=2),
            data,
            "lanes",
            LANE_MIN_SPEEDUP,
        ),
        "lane_feature_std": Workload(
            "lane_feature_std",
            temporal_features.std(axis=2),
            data,
            "lanes",
            LANE_MIN_SPEEDUP,
        ),
    }


def verify_equal(reference: Path, candidate: Path) -> tuple[float, float]:
    reference_size = reference.stat().st_size
    candidate_size = candidate.stat().st_size
    if candidate_size != reference_size:
        raise RuntimeError(
            f"output size mismatch: {candidate_size} != {reference_size}"
        )
    count = reference_size // np.dtype(np.float64).itemsize
    left = np.memmap(reference, mode="r", dtype=np.float64, shape=(count,))
    right = np.memmap(candidate, mode="r", dtype=np.float64, shape=(count,))
    checksum = 0.0
    finite_count = 0
    for start in range(0, count, 1_048_576):
        stop = min(start + 1_048_576, count)
        left_chunk = np.asarray(left[start:stop])
        right_chunk = np.asarray(right[start:stop])
        left_nan = np.isnan(left_chunk)
        right_nan = np.isnan(right_chunk)
        if not np.array_equal(left_nan, right_nan):
            raise RuntimeError(f"NaN-mask mismatch at flat slice [{start}:{stop}]")
        finite = ~left_nan
        if not np.array_equal(left_chunk[finite], right_chunk[finite]):
            raise RuntimeError(f"value mismatch at flat slice [{start}:{stop}]")
        checksum += float(np.sum(right_chunk[finite], dtype=np.float64))
        finite_count += int(np.count_nonzero(finite))
    del left, right
    finite_fraction = finite_count / count if count else 1.0
    return checksum, finite_fraction


def benchmark(workload: Workload, counts: tuple[int, ...]) -> list[dict[str, object]]:
    runtime = compile_formula(
        workload.formula,
        workload.data,
        n_instruments=N,
        prefetch_rows=16,
    )
    if runtime.parallel_plan.mode != workload.expected_mode:
        raise RuntimeError(
            f"{workload.name} planned as {runtime.parallel_plan.mode!r}, "
            f"expected {workload.expected_mode!r}: {runtime.parallel_plan.reason}"
        )

    outputs = {
        threads: OUTPUT_DIR / f"parallel_reduction_{workload.name}_{threads}.bin"
        for threads in counts
    }
    for threads in counts:
        for _ in range(WARMUPS):
            runtime.run(
                out_path=outputs[threads],
                threads=threads,
                pin_threads=True,
                async_writeback_mb=0,
            )

    measured: dict[int, list[object]] = {threads: [] for threads in counts}
    forward = counts
    backward = tuple(reversed(counts))
    for repetition in range(RUNS):
        order = forward if repetition % 2 == 0 else backward
        for threads in order:
            measured[threads].append(
                runtime.run(
                    out_path=outputs[threads],
                    threads=threads,
                    pin_threads=True,
                    async_writeback_mb=0,
                )
            )

    reference = outputs[1]
    checksums: dict[int, float] = {}
    finite_fractions: dict[int, float] = {}
    reference_count = reference.stat().st_size // np.dtype(np.float64).itemsize
    reference_values = np.memmap(
        reference, mode="r", dtype=np.float64, shape=(reference_count,)
    )
    reference_finite = np.isfinite(reference_values)
    checksums[1] = float(
        np.sum(reference_values[reference_finite], dtype=np.float64)
    )
    finite_fractions[1] = float(np.mean(reference_finite))
    del reference_values
    for threads in counts[1:]:
        checksums[threads], finite_fractions[threads] = verify_equal(
            reference, outputs[threads]
        )

    rows: list[dict[str, object]] = []
    baseline_rate = median(
        run.rows_per_second for run in measured[1]
    )
    for requested in counts:
        runs = measured[requested]
        rates = [run.rows_per_second for run in runs]
        busy = [run.average_busy_cores for run in runs]
        actual_threads = runs[0].threads
        if requested == 1 and actual_threads != 1:
            raise RuntimeError(
                f"serial request used {actual_threads} threads for {workload.name}"
            )
        if requested > 1 and actual_threads < 2:
            raise RuntimeError(
                f"parallel request stayed serial for {workload.name}: "
                f"requested={requested} actual={actual_threads}"
            )
        rate = median(rates)
        rows.append(
            {
                "case": workload.name,
                "mode": runtime.parallel_plan.mode,
                "reason": runtime.parallel_plan.reason,
                "requested": requested,
                "actual": actual_threads,
                "median_rate": rate,
                "mean_rate": mean(rates),
                "best_rate": max(rates),
                "median_seconds": median(run.seconds for run in runs),
                "median_busy": median(busy),
                "speedup": rate / baseline_rate,
                "efficiency": (rate / baseline_rate) / actual_threads,
                "rates": rates,
                "busy": busy,
                "checksum": checksums[requested],
                "finite_fraction": finite_fractions[requested],
                "output_bytes": outputs[requested].stat().st_size,
            }
        )

    for multicore in rows[1:]:
        if float(multicore["speedup"]) < workload.minimum_speedup:
            raise RuntimeError(
                f"{workload.name} requested_threads={multicore['requested']} "
                f"did not meet speedup floor: {multicore['speedup']:.4f}x < "
                f"{workload.minimum_speedup:.4f}x"
            )
        if float(multicore["median_busy"]) <= 1.10:
            raise RuntimeError(
                f"{workload.name} requested_threads={multicore['requested']} "
                f"did not demonstrate multicore execution: "
                f"median_busy_cores={multicore['median_busy']:.3f}"
            )

    for path in outputs.values():
        path.unlink(missing_ok=True)
    return rows


def main() -> None:
    cpu_ids = available_cpu_ids()
    if len(cpu_ids) < 2:
        raise SystemExit(
            f"parallel reduction benchmark requires multiple CPUs; affinity={cpu_ids}"
        )
    counts = thread_counts(len(cpu_ids))
    cases = selected_cases()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"available_cpus={len(cpu_ids)} cpu_ids={cpu_ids}")
    print(f"thread_counts={counts} pin_threads=True async_writeback_mb=0")
    print(f"rows={ROWS:,} instruments={N} warmups={WARMUPS} runs={RUNS}")
    print(
        f"row_min_speedup={ROW_MIN_SPEEDUP:.3f}x "
        f"lane_min_speedup={LANE_MIN_SPEEDUP:.3f}x"
    )
    print("measurement_order=alternating_forward_reverse")

    with tempfile.TemporaryDirectory(
        prefix="cpp_stream_parallel_reductions_"
    ) as temporary:
        workloads = build_workloads(Path(temporary))
        all_results: list[dict[str, object]] = []
        for name in cases:
            all_results.extend(benchmark(workloads[name], counts))

    for result in all_results:
        print("---")
        print(f"case={result['case']}")
        print(f"mode={result['mode']} reason={result['reason']}")
        print(
            f"requested_threads={result['requested']} "
            f"actual_threads={result['actual']}"
        )
        print(f"median={float(result['median_rate']) / 1e6:.6f} M rows/s")
        print(f"mean={float(result['mean_rate']) / 1e6:.6f} M rows/s")
        print(f"best={float(result['best_rate']) / 1e6:.6f} M rows/s")
        print(f"median_seconds={float(result['median_seconds']):.6f}")
        print(f"speedup={float(result['speedup']):.4f}x")
        print(f"parallel_efficiency={100.0 * float(result['efficiency']):.2f}%")
        print(f"median_busy_cores={float(result['median_busy']):.3f}")
        print(f"output_bytes={result['output_bytes']}")
        print(
            "runs="
            + ", ".join(
                f"{float(rate) / 1e6:.6f}" for rate in result["rates"]
            )
            + " M rows/s"
        )
        print(
            "busy_cores="
            + ", ".join(f"{float(value):.3f}" for value in result["busy"])
        )
        print(f"checksum={float(result['checksum']):.12g}")
        print(f"finite_fraction={float(result['finite_fraction']):.12g}")


if __name__ == "__main__":
    main()
