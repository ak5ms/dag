from __future__ import annotations

import os
from pathlib import Path
from statistics import mean, median
import tempfile

import numpy as np

from trading_dsl_engine.cpp_stream import compile_formula

ROWS = int(os.environ.get("CPP_STREAM_BENCH_ROWS", "5000000"))
N = int(os.environ.get("CPP_STREAM_BENCH_INSTRUMENTS", "9"))
RUNS = int(os.environ.get("CPP_STREAM_BENCH_RUNS", "10"))
WARMUPS = int(os.environ.get("CPP_STREAM_BENCH_WARMUPS", "1"))
PREFETCH_ROWS = int(os.environ.get("CPP_STREAM_BENCH_PREFETCH_ROWS", "16"))
MIN_MROWS = float(os.environ.get("CPP_STREAM_BENCH_MIN_MROWS", "0"))


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="cpp_stream_bench_") as tmp:
        root = Path(tmp)
        rng = np.random.default_rng(42)
        paths = {}
        for name in ("close", "open"):
            path = root / f"{name}.bin"
            arr = np.memmap(path, mode="w+", dtype=np.float64, shape=(ROWS, N))
            for start in range(0, ROWS, 131072):
                stop = min(start + 131072, ROWS)
                arr[start:stop] = rng.lognormal(4.0, 0.12, (stop - start, N))
            arr.flush()
            del arr
            paths[name] = path

        runtime = compile_formula(
            "xs_rank(ewm(close / open, 21))",
            n_instruments=N,
            prefetch_rows=PREFETCH_ROWS,
        )
        out = root / "out.bin"

        # MMapFile no longer truncates an already-correct output size. Warmups
        # therefore remove first-allocation/page-fault noise without changing the
        # measured row loop or requiring an in-memory output implementation.
        for _ in range(WARMUPS):
            runtime.run_files(paths, out_path=out, async_writeback_mb=0)

        rates = [
            runtime.run_files(paths, out_path=out, async_writeback_mb=0).rows_per_second
            for _ in range(RUNS)
        ]
        median_mrows = median(rates) / 1e6
        print(f"rows={ROWS:,} instruments={N} warmups={WARMUPS} runs={RUNS}")
        print(f"prefetch_rows={PREFETCH_ROWS}")
        print(f"median={median_mrows:.3f} M rows/s")
        print(f"mean={mean(rates) / 1e6:.3f} M rows/s")
        print(f"best={max(rates) / 1e6:.3f} M rows/s")
        print("runs=" + ", ".join(f"{rate / 1e6:.3f}" for rate in rates) + " M rows/s")
        print(f"generated_cpp={runtime.generated_cpp}")

        if MIN_MROWS > 0 and median_mrows < MIN_MROWS:
            raise SystemExit(
                f"cpp_stream regression: median {median_mrows:.3f} M rows/s "
                f"is below CPP_STREAM_BENCH_MIN_MROWS={MIN_MROWS:.3f}"
            )


if __name__ == "__main__":
    main()
