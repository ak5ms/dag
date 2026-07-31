from __future__ import annotations

import os
from pathlib import Path
from statistics import median
import tempfile

import numpy as np

from trading_dsl_engine.cpp_stream import compile_formula

ROWS = int(os.environ.get("CPP_STREAM_BENCH_ROWS", "5000000"))
N = int(os.environ.get("CPP_STREAM_BENCH_INSTRUMENTS", "9"))
RUNS = int(os.environ.get("CPP_STREAM_BENCH_RUNS", "10"))


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

        runtime = compile_formula("xs_rank(ewm(close / open, 21))", n_instruments=N, prefetch_rows=16)
        out = root / "out.bin"
        runtime.run_files(paths, out_path=out)
        rates = []
        for _ in range(RUNS):
            rates.append(runtime.run_files(paths, out_path=out).rows_per_second)
        print(f"rows={ROWS:,} instruments={N} runs={RUNS}")
        print(f"median={median(rates) / 1e6:.3f} M rows/s")
        print(f"best={max(rates) / 1e6:.3f} M rows/s")
        print("runs=" + ", ".join(f"{x / 1e6:.3f}" for x in rates) + " M rows/s")


if __name__ == "__main__":
    main()
