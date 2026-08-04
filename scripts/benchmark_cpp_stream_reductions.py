from __future__ import annotations

import os
from pathlib import Path
from statistics import median
from time import perf_counter
import tempfile

import numpy as np

from trading_dsl_engine.base.dsl import cat, cumsum, var
from trading_dsl_engine.cpp_stream import compile_formula


ROWS = int(os.environ.get("CPP_STREAM_REDUCTION_ROWS", "5000000"))
N = int(os.environ.get("CPP_STREAM_REDUCTION_INSTRUMENTS", "9"))
RUNS = int(os.environ.get("CPP_STREAM_REDUCTION_RUNS", "10"))
WARMUPS = int(os.environ.get("CPP_STREAM_REDUCTION_WARMUPS", "1"))
OUTPUT_ROOT = Path(os.environ.get("CPP_STREAM_BENCH_OUTPUT_DIR", tempfile.gettempdir()))


def rates(runtime, output: Path):
    for _ in range(WARMUPS):
        runtime.run(out_path=output, async_writeback_mb=0)
    results = [runtime.run(out_path=output, async_writeback_mb=0) for _ in range(RUNS)]
    return results, [result.rows_per_second for result in results]


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="cpp_stream_reduction_") as temporary:
        root = Path(temporary)
        rng = np.random.default_rng(42)
        paths = {name: root / f"{name}.npy" for name in ("x", "y")}
        for name in paths:
            array = np.lib.format.open_memmap(
                paths[name], mode="w+", dtype=np.float64, shape=(ROWS, N)
            )
            for start in range(0, ROWS, 131072):
                stop = min(start + 131072, ROWS)
                array[start:stop] = rng.normal(size=(stop - start, N))
            array.flush()
            del array

        x = var("x")
        y = var("y")
        computation = cat(x * 1.01 + y, x - y * 0.1, x * y)
        full = compile_formula(computation, paths, n_instruments=N)
        reduced = compile_formula(computation.sum(axis=0), paths, n_instruments=N)
        mean_runtime = compile_formula(computation.mean(axis=0), paths, n_instruments=N)
        std_runtime = compile_formula(computation.std(axis=0), paths, n_instruments=N)
        x_sum_runtime = compile_formula(
            x.sum(axis=0), {"x": paths["x"]}, n_instruments=N
        )
        emit_runtime = compile_formula(
            cumsum(x).emit("last"), {"x": paths["x"]}, n_instruments=N
        )

        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        full_path = OUTPUT_ROOT / "cpp_stream_reduction_full.bin"
        reduced_path = OUTPUT_ROOT / "cpp_stream_reduction_sum.bin"
        mean_path = OUTPUT_ROOT / "cpp_stream_reduction_mean.bin"
        std_path = OUTPUT_ROOT / "cpp_stream_reduction_std.bin"
        x_sum_path = OUTPUT_ROOT / "cpp_stream_reduction_x_sum.bin"
        emit_path = OUTPUT_ROOT / "cpp_stream_reduction_emit.bin"

        full_results, full_rates = rates(full, full_path)
        reduced_results, reduced_rates = rates(reduced, reduced_path)
        _, mean_rates = rates(mean_runtime, mean_path)
        _, std_rates = rates(std_runtime, std_path)
        x_sum_results, x_sum_rates = rates(x_sum_runtime, x_sum_path)
        emit_results, emit_rates = rates(emit_runtime, emit_path)

        started = perf_counter()
        materialized = np.memmap(
            full_path, mode="r", dtype=np.float64, shape=(ROWS, N, 3)
        )
        post_sum = np.nansum(materialized, axis=0)
        post_seconds = perf_counter() - started
        native_sum = np.fromfile(reduced_path, dtype=np.float64).reshape(N, 3)
        np.testing.assert_allclose(native_sum, post_sum, rtol=1e-11, atol=1e-8)

        x_sum = np.fromfile(x_sum_path, dtype=np.float64).reshape(N)
        emit_last = np.fromfile(emit_path, dtype=np.float64).reshape(N)
        np.testing.assert_allclose(emit_last, x_sum, rtol=1e-12, atol=1e-9)

        full_median = median(full_rates)
        reduced_median = median(reduced_rates)
        if reduced_median <= full_median:
            raise RuntimeError(
                f"streaming reduction was not faster: {reduced_median=} {full_median=}"
            )

        full_bytes = full_path.stat().st_size
        reduced_bytes = reduced_path.stat().st_size
        full_seconds = median(result.seconds for result in full_results)
        reduced_seconds = median(result.seconds for result in reduced_results)
        x_sum_seconds = median(result.seconds for result in x_sum_results)
        emit_seconds = median(result.seconds for result in emit_results)

        print(f"rows={ROWS:,} instruments={N} features=3 warmups={WARMUPS} runs={RUNS}")
        print(f"full_median={full_median/1e6:.6f} M rows/s seconds={full_seconds:.6f} bytes={full_bytes}")
        print(f"sum_axis0_median={reduced_median/1e6:.6f} M rows/s seconds={reduced_seconds:.6f} bytes={reduced_bytes}")
        print(f"native_reduction_speedup={reduced_median/full_median:.3f}x")
        print(f"full_plus_numpy_reduction_seconds={full_seconds + post_seconds:.6f}")
        print(f"native_vs_full_plus_post_speedup={(full_seconds + post_seconds)/reduced_seconds:.3f}x")
        print(f"mean_axis0_median={median(mean_rates)/1e6:.6f} M rows/s")
        print(f"std_axis0_median={median(std_rates)/1e6:.6f} M rows/s")
        print(f"x_sum_axis0_median={median(x_sum_rates)/1e6:.6f} M rows/s seconds={x_sum_seconds:.6f}")
        print(f"cumsum_emit_last_median={median(emit_rates)/1e6:.6f} M rows/s seconds={emit_seconds:.6f}")
        print(f"emit_vs_equivalent_sum_speedup={x_sum_seconds/emit_seconds:.3f}x")
        print(f"checksum={float(np.nansum(native_sum)):.12g}")
        print(f"x_checksum={float(np.nansum(x_sum)):.12g}")

        for output in (
            full_path,
            reduced_path,
            mean_path,
            std_path,
            x_sum_path,
            emit_path,
        ):
            output.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
