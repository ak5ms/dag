from __future__ import annotations

import os
from pathlib import Path
from statistics import mean, median
import tempfile

import numpy as np

from trading_dsl_engine.cpp_stream import compile_formula


ROWS = int(os.environ.get("CPP_STREAM_XS_GAUSS_ROWS", "2000000"))
N = int(os.environ.get("CPP_STREAM_XS_GAUSS_INSTRUMENTS", "9"))
RUNS = int(os.environ.get("CPP_STREAM_XS_GAUSS_RUNS", "7"))
WARMUPS = int(os.environ.get("CPP_STREAM_XS_GAUSS_WARMUPS", "1"))
PREFETCH_ROWS = int(os.environ.get("CPP_STREAM_XS_GAUSS_PREFETCH_ROWS", "16"))
OUTPUT_DIR = os.environ.get("CPP_STREAM_BENCH_OUTPUT_DIR")


def _build_input(path: Path) -> None:
    array = np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=np.float64,
        shape=(ROWS, N),
    )
    lane = np.arange(N, dtype=np.float64)[None, :]
    chunk = 131_072
    for start in range(0, ROWS, chunk):
        stop = min(start + chunk, ROWS)
        row = np.arange(start, stop, dtype=np.float64)[:, None]
        values = (
            np.sin(0.00017 * row + 0.31 * lane)
            + 0.35 * np.cos(0.000031 * row - 0.19 * lane)
            + 0.025 * lane
        )
        # Exercise the formerly explosive branch without materially changing the
        # benchmark mix: one nearly common nonzero row per 4096 observations.
        local = np.arange(start, stop)
        nearly_common = (local % 4096) == 0
        if np.any(nearly_common):
            values[nearly_common] = 1.0 + 1e-12 * (lane - 0.5 * (N - 1))
        array[start:stop] = values
    array.flush()
    del array


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="cpp_stream_xs_gauss_") as temporary:
        root = Path(temporary)
        input_path = root / "x.npy"
        _build_input(input_path)
        runtime = compile_formula(
            "xs_gauss(x)",
            {"x": input_path},
            n_instruments=N,
            prefetch_rows=PREFETCH_ROWS,
        )
        output_root = Path(OUTPUT_DIR) if OUTPUT_DIR else root
        output_root.mkdir(parents=True, exist_ok=True)
        output = output_root / f"cpp_stream_xs_gauss_{N}.bin"

        for _ in range(WARMUPS):
            runtime.run(out_path=output, async_writeback_mb=0)
        rates = [
            runtime.run(out_path=output, async_writeback_mb=0).rows_per_second
            for _ in range(RUNS)
        ]
        values = np.memmap(output, mode="r", dtype=np.float64, shape=(ROWS, N))
        tail = values[-min(8192, ROWS):]
        checksum = float(np.nansum(tail))
        finite_fraction = float(np.isfinite(tail).mean())
        max_abs = float(np.nanmax(np.abs(tail)))
        del values

        print("formula=xs_gauss(x)")
        print(f"rows={ROWS:,} instruments={N} warmups={WARMUPS} runs={RUNS}")
        print(f"median={median(rates) / 1e6:.6f} M rows/s")
        print(f"mean={mean(rates) / 1e6:.6f} M rows/s")
        print(f"best={max(rates) / 1e6:.6f} M rows/s")
        print("runs=" + ", ".join(f"{rate / 1e6:.6f}" for rate in rates) + " M rows/s")
        print(f"checksum={checksum:.12g}")
        print(f"tail_finite_fraction={finite_fraction:.12g}")
        print(f"tail_max_abs={max_abs:.12g}")
        print(f"generated_cpp={runtime.generated_cpp}")


if __name__ == "__main__":
    main()
