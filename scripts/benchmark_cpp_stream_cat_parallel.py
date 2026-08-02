from __future__ import annotations

import os
from pathlib import Path
from statistics import mean, median
import tempfile

import numpy as np

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


def _sample(path: Path, output_row_width: int) -> np.ndarray:
    values = np.memmap(
        path,
        mode="r",
        dtype=np.float64,
        shape=(ROWS, output_row_width),
    )
    result = np.asarray(values[-min(1024, ROWS) :]).copy()
    del values
    return result


def _run_mode(runtime, output: Path, *, threads: int) -> dict[str, object]:
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
    sample = _sample(output, runtime.plan.output_row_width)

    # Cat reads three N-wide float64 inputs and writes one N x 3 float64 output.
    # This is the minimum payload traffic and excludes cache-line write allocate,
    # page-table traffic, and output writeback performed outside the measured loop.
    bytes_per_row = 6 * N * np.dtype(np.float64).itemsize
    return {
        "rates": rates,
        "busy": busy,
        "threads": actual_threads,
        "available": available,
        "sample": sample,
        "median_mrows": median(rates) / 1e6,
        "mean_mrows": mean(rates) / 1e6,
        "best_mrows": max(rates) / 1e6,
        "median_busy": median(busy),
        "minimum_gbs": median(rates) * bytes_per_row / 1e9,
    }


def _print_mode(name: str, result: dict[str, object]) -> None:
    rates = result["rates"]
    busy = result["busy"]
    print("---")
    print(f"mode={name}")
    print(f"threads={result['threads']}")
    print(f"available_cpus={result['available']}")
    print(f"median={result['median_mrows']:.3f} M rows/s")
    print(f"mean={result['mean_mrows']:.3f} M rows/s")
    print(f"best={result['best_mrows']:.3f} M rows/s")
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

        runtime = compile_formula(
            "cat(x1, x2, x3)",
            paths,
            n_instruments=N,
            prefetch_rows=PREFETCH_ROWS,
        )
        if runtime.parallel_plan.mode != "rows":
            raise AssertionError(
                f"Cat root should be row-parallel, got {runtime.parallel_plan}"
            )
        if not runtime.parallel_plan.auto_multicore:
            raise AssertionError("Cat root should automatically use multiple CPUs")

        output_root = Path(OUTPUT_DIR) if OUTPUT_DIR else root
        output_root.mkdir(parents=True, exist_ok=True)
        serial_path = output_root / "cpp_stream_cat_serial.bin"
        automatic_path = output_root / "cpp_stream_cat_automatic.bin"

        print(f"rows={ROWS:,} instruments={N} features=3")
        print(f"warmups={WARMUPS} runs={RUNS} affinity_cpus={affinity_cpus}")
        print(f"parallel_plan={runtime.parallel_plan}")
        print("parallelization=whole-plan row sharding; no nested Cat task pool")

        serial = _run_mode(runtime, serial_path, threads=1)
        automatic = _run_mode(runtime, automatic_path, threads=0)
        np.testing.assert_array_equal(serial["sample"], automatic["sample"])

        automatic_threads = automatic["threads"]
        if REQUIRE_MULTICORE and (
            len(automatic_threads) != 1 or automatic_threads[0] < 2
        ):
            raise AssertionError(
                "automatic Cat execution did not use multiple threads: "
                f"{automatic_threads}"
            )

        _print_mode("serial", serial)
        _print_mode("automatic", automatic)
        speedup = automatic["median_mrows"] / serial["median_mrows"]
        print("---")
        print(f"speedup={speedup:.3f}x")
        print(f"checksum={float(np.sum(automatic['sample'])):.12g}")
        print(f"generated_cpp={runtime.generated_cpp}")

        serial_path.unlink(missing_ok=True)
        automatic_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
