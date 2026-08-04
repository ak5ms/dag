from __future__ import annotations

import os
from pathlib import Path
from statistics import mean, median
from time import perf_counter
import tempfile

import numpy as np

from trading_dsl_engine.base.dsl import cat, var
from trading_dsl_engine.cpp_stream import compile_formula


ROWS = int(os.environ.get("CPP_STREAM_CAT_ROWS", "5000000"))
N = int(os.environ.get("CPP_STREAM_CAT_INSTRUMENTS", "9"))
RUNS = int(os.environ.get("CPP_STREAM_CAT_RUNS", "10"))
WARMUPS = int(os.environ.get("CPP_STREAM_CAT_WARMUPS", "1"))
OUTPUT_DIR = os.environ.get("CPP_STREAM_BENCH_OUTPUT_DIR")
PREFETCH_ROWS = int(os.environ.get("CPP_STREAM_BENCH_PREFETCH_ROWS", "16"))
PIN_THREADS = os.environ.get("CPP_STREAM_PIN_THREADS", "1").lower() not in {
    "0",
    "false",
    "no",
    "off",
}
REQUIRE_MULTICORE = os.environ.get(
    "CPP_STREAM_REQUIRE_MULTICORE", "1"
).lower() not in {"0", "false", "no", "off"}


def _available_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def _write_npy(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    values = np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=np.float64,
        shape=(ROWS, N),
    )
    chunk = 131_072
    for start in range(0, ROWS, chunk):
        stop = min(start + chunk, ROWS)
        values[start:stop] = rng.normal(size=(stop - start, N))
    values.flush()
    del values


def _sample(
    path: Path, output_row_width: int, output_rows: int
) -> np.ndarray:
    values = np.memmap(
        path,
        mode="r",
        dtype=np.float64,
        shape=(output_rows, output_row_width),
    )
    result = np.asarray(values[-min(1024, output_rows) :]).copy()
    del values
    return result


def _run_mode(
    runtime,
    output: Path,
    *,
    threads: int,
    payload_bytes_per_row: int,
) -> dict[str, object]:
    for _ in range(WARMUPS):
        runtime.run(
            out_path=output,
            threads=threads,
            pin_threads=PIN_THREADS,
            async_writeback_mb=0,
        )

    results = [
        runtime.run(
            out_path=output,
            threads=threads,
            pin_threads=PIN_THREADS,
            async_writeback_mb=0,
        )
        for _ in range(RUNS)
    ]
    rates = [result.rows_per_second for result in results]
    busy = [result.average_busy_cores for result in results]
    actual_threads = sorted({result.threads for result in results})
    available = sorted({result.available_cpus for result in results})
    output_rows = sorted({result.output_rows for result in results})
    if len(output_rows) != 1:
        raise AssertionError(f"inconsistent output row counts: {output_rows}")
    sample = _sample(
        output,
        runtime.plan.output_row_width,
        output_rows[0],
    )

    return {
        "rates": rates,
        "busy": busy,
        "threads": actual_threads,
        "available": available,
        "output_rows": output_rows[0],
        "output_bytes": output.stat().st_size,
        "sample": sample,
        "median_mrows": median(rates) / 1e6,
        "mean_mrows": mean(rates) / 1e6,
        "best_mrows": max(rates) / 1e6,
        "median_seconds": median(result.seconds for result in results),
        "median_busy": median(busy),
        "minimum_gbs": median(rates) * payload_bytes_per_row / 1e9,
    }


def _print_mode(name: str, result: dict[str, object]) -> None:
    rates = result["rates"]
    busy = result["busy"]
    print("---")
    print(f"mode={name}")
    print(f"threads={result['threads']}")
    print(f"available_cpus={result['available']}")
    print(f"output_rows={result['output_rows']}")
    print(f"output_bytes={result['output_bytes']}")
    print(f"median={result['median_mrows']:.3f} M rows/s")
    print(f"mean={result['mean_mrows']:.3f} M rows/s")
    print(f"best={result['best_mrows']:.3f} M rows/s")
    print(f"median_seconds={result['median_seconds']:.6f}")
    print(f"median_busy_cores={result['median_busy']:.3f}")
    print(f"minimum_payload_bandwidth={result['minimum_gbs']:.3f} GB/s")
    print(
        "runs="
        + ", ".join(f"{rate / 1e6:.3f}" for rate in rates)
        + " M rows/s"
    )
    print("busy_cores=" + ", ".join(f"{value:.3f}" for value in busy))


def main() -> None:
    if ROWS <= 0 or N <= 0 or RUNS <= 0 or WARMUPS < 0:
        raise ValueError("rows, instruments, and runs must be positive; warmups >= 0")
    affinity_cpus = _available_cpus()
    if REQUIRE_MULTICORE and affinity_cpus < 2:
        raise SystemExit(
            "Cat multicore benchmark requires at least two CPUs in the process "
            f"affinity mask; found {affinity_cpus}"
        )

    with tempfile.TemporaryDirectory(prefix="cpp_stream_cat_parallel_") as temporary:
        root = Path(temporary)
        paths = {name: root / f"{name}.npy" for name in ("x1", "x2", "x3")}
        for seed, path in enumerate(paths.values(), start=1):
            _write_npy(path, seed)

        features = cat(var("x1"), var("x2"), var("x3"))
        full_runtime = compile_formula(
            features,
            paths,
            n_instruments=N,
            prefetch_rows=PREFETCH_ROWS,
        )
        feature_sum_runtime = compile_formula(
            features.sum(axis=2),
            paths,
            n_instruments=N,
            prefetch_rows=PREFETCH_ROWS,
        )
        all_sum_runtime = compile_formula(
            features.sum(),
            paths,
            n_instruments=N,
            prefetch_rows=PREFETCH_ROWS,
        )
        for name, runtime in (
            ("full Cat", full_runtime),
            ("feature-axis sum", feature_sum_runtime),
        ):
            if runtime.parallel_plan.mode != "rows":
                raise AssertionError(
                    f"{name} should be row-parallel, got {runtime.parallel_plan}"
                )
            if not runtime.parallel_plan.auto_multicore:
                raise AssertionError(f"{name} should automatically use multiple CPUs")
        if all_sum_runtime.plan.output_mode != "final":
            raise AssertionError("axis-free Cat sum should emit one final scalar")

        output_root = Path(OUTPUT_DIR) if OUTPUT_DIR else root
        output_root.mkdir(parents=True, exist_ok=True)
        paths_out = {
            "full_serial": output_root / "cpp_stream_cat_serial.bin",
            "full_automatic": output_root / "cpp_stream_cat_automatic.bin",
            "feature_serial": output_root / "cpp_stream_cat_feature_sum_serial.bin",
            "feature_automatic": output_root
            / "cpp_stream_cat_feature_sum_automatic.bin",
            "all_sum": output_root / "cpp_stream_cat_all_sum.bin",
        }

        scalar_bytes = np.dtype(np.float64).itemsize
        full_payload_bytes = 6 * N * scalar_bytes
        feature_sum_payload_bytes = 4 * N * scalar_bytes
        all_sum_payload_bytes = 3 * N * scalar_bytes

        print(f"rows={ROWS:,} instruments={N} features=3")
        print(f"warmups={WARMUPS} runs={RUNS} affinity_cpus={affinity_cpus}")
        print(f"full_parallel_plan={full_runtime.parallel_plan}")
        print(f"feature_sum_parallel_plan={feature_sum_runtime.parallel_plan}")
        print(f"all_sum_parallel_plan={all_sum_runtime.parallel_plan}")
        print("parallelization=whole-plan sharding; no nested Cat task pool")

        full_serial = _run_mode(
            full_runtime,
            paths_out["full_serial"],
            threads=1,
            payload_bytes_per_row=full_payload_bytes,
        )
        full_automatic = _run_mode(
            full_runtime,
            paths_out["full_automatic"],
            threads=0,
            payload_bytes_per_row=full_payload_bytes,
        )
        feature_serial = _run_mode(
            feature_sum_runtime,
            paths_out["feature_serial"],
            threads=1,
            payload_bytes_per_row=feature_sum_payload_bytes,
        )
        feature_automatic = _run_mode(
            feature_sum_runtime,
            paths_out["feature_automatic"],
            threads=0,
            payload_bytes_per_row=feature_sum_payload_bytes,
        )
        all_sum = _run_mode(
            all_sum_runtime,
            paths_out["all_sum"],
            threads=1,
            payload_bytes_per_row=all_sum_payload_bytes,
        )

        np.testing.assert_array_equal(
            full_serial["sample"], full_automatic["sample"]
        )
        np.testing.assert_array_equal(
            feature_serial["sample"], feature_automatic["sample"]
        )
        full_tail = full_automatic["sample"].reshape(-1, N, 3)
        np.testing.assert_allclose(
            feature_automatic["sample"],
            np.sum(full_tail, axis=2),
            rtol=1e-13,
            atol=1e-13,
        )

        started = perf_counter()
        materialized = np.memmap(
            paths_out["full_serial"],
            mode="r",
            dtype=np.float64,
            shape=(ROWS, N, 3),
        )
        post_sum = float(np.sum(materialized, dtype=np.float64))
        post_seconds = perf_counter() - started
        del materialized
        native_sum = float(all_sum["sample"].reshape(-1)[0])
        np.testing.assert_allclose(
            native_sum,
            post_sum,
            rtol=1e-11,
            atol=1e-7,
        )

        automatic_threads = full_automatic["threads"]
        if REQUIRE_MULTICORE and (
            len(automatic_threads) != 1 or automatic_threads[0] < 2
        ):
            raise AssertionError(
                "automatic Cat execution did not use multiple threads: "
                f"{automatic_threads}"
            )

        _print_mode("full Cat serial", full_serial)
        _print_mode("full Cat automatic", full_automatic)
        _print_mode("Cat feature-axis sum serial", feature_serial)
        _print_mode("Cat feature-axis sum automatic", feature_automatic)
        _print_mode("Cat all-axis sum", all_sum)

        feature_serial_speedup = (
            feature_serial["median_mrows"] / full_serial["median_mrows"]
        )
        feature_automatic_speedup = (
            feature_automatic["median_mrows"]
            / full_automatic["median_mrows"]
        )
        all_sum_speedup = all_sum["median_mrows"] / full_serial["median_mrows"]
        if feature_serial_speedup <= 1.0 or feature_automatic_speedup <= 1.0:
            raise RuntimeError(
                "feature reduction was not faster than full Cat output: "
                f"serial={feature_serial_speedup:.3f}x "
                f"automatic={feature_automatic_speedup:.3f}x"
            )
        if all_sum_speedup <= 1.0:
            raise RuntimeError(
                "terminal reduction was not faster than serial full Cat output: "
                f"{all_sum_speedup:.3f}x"
            )

        print("---")
        print(
            f"full_cat_multicore_speedup="
            f"{full_automatic['median_mrows'] / full_serial['median_mrows']:.3f}x"
        )
        print(f"feature_sum_serial_vs_full={feature_serial_speedup:.3f}x")
        print(
            f"feature_sum_automatic_vs_full={feature_automatic_speedup:.3f}x"
        )
        print(f"all_sum_vs_full_serial={all_sum_speedup:.3f}x")
        print(f"post_hoc_sum_seconds={post_seconds:.6f}")
        print(
            "all_sum_vs_full_plus_post="
            f"{(full_serial['median_seconds'] + post_seconds) / all_sum['median_seconds']:.3f}x"
        )
        print(f"checksum={native_sum:.12g}")
        print(f"generated_cpp_full={full_runtime.generated_cpp}")
        print(f"generated_cpp_feature_sum={feature_sum_runtime.generated_cpp}")
        print(f"generated_cpp_all_sum={all_sum_runtime.generated_cpp}")

        for output in paths_out.values():
            output.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
