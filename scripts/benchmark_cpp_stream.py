from __future__ import annotations

import os
from pathlib import Path
from statistics import mean, median
import tempfile

import numpy as np

from trading_dsl_engine.base.dsl import cumsum, ewm, groupby, self_, univ, var
from trading_dsl_engine.cpp_stream import compile_formula

ROWS = int(os.environ.get("CPP_STREAM_BENCH_ROWS", "5000000"))
N = int(os.environ.get("CPP_STREAM_BENCH_INSTRUMENTS", "9"))
RUNS = int(os.environ.get("CPP_STREAM_BENCH_RUNS", "10"))
WARMUPS = int(os.environ.get("CPP_STREAM_BENCH_WARMUPS", "1"))
PREFETCH_ROWS = int(os.environ.get("CPP_STREAM_BENCH_PREFETCH_ROWS", "16"))
MIN_MROWS = float(os.environ.get("CPP_STREAM_BENCH_MIN_MROWS", "0"))
CASE = os.environ.get("CPP_STREAM_BENCH_CASE", "minute_groupby")


def _write_matrix(path: Path, values, *, chunk_rows: int = 131072) -> None:
    arr = np.memmap(path, mode="w+", dtype=np.float64, shape=(ROWS, N))
    for start in range(0, ROWS, chunk_rows):
        stop = min(start + chunk_rows, ROWS)
        arr[start:stop] = values(start, stop)
    arr.flush()
    del arr


def _build_case(root: Path):
    rng = np.random.default_rng(42)
    close_path = root / "close.bin"
    _write_matrix(
        close_path,
        lambda start, stop: rng.lognormal(4.0, 0.12, (stop - start, N)),
    )

    if CASE == "minute_groupby":
        if N != 9:
            raise ValueError("minute_groupby benchmark uses the requested 9-column univ partition")
        timestamp_path = root / "_ev_ts.bin"
        day_us = 86_400_000_000.0
        minute_us = 60_000_000.0
        base = np.floor(1_700_000_000_000_000.0 / day_us) * day_us

        def timestamps(start: int, stop: int) -> np.ndarray:
            row_ts = base + np.arange(start, stop, dtype=np.float64) * minute_us
            return np.broadcast_to(row_ts[:, None], (stop - start, N))

        _write_matrix(timestamp_path, timestamps)
        formula = groupby(
            (univ([0], [1, 2], list(range(3, 9))), var("minute")),
            var("close"),
            ewm(cumsum(self_), 3),
        )
        paths = {"_ev_ts": timestamp_path, "close": close_path}
        return formula, paths

    if CASE == "rank":
        open_path = root / "open.bin"
        _write_matrix(
            open_path,
            lambda start, stop: rng.lognormal(4.0, 0.12, (stop - start, N)),
        )
        return "xs_rank(ewm(close / open, 21))", {"close": close_path, "open": open_path}

    raise ValueError(f"unknown CPP_STREAM_BENCH_CASE={CASE!r}")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="cpp_stream_bench_") as tmp:
        root = Path(tmp)
        formula, paths = _build_case(root)
        runtime = compile_formula(
            formula,
            n_instruments=N,
            prefetch_rows=PREFETCH_ROWS,
        )
        out = root / "out.bin"

        for _ in range(WARMUPS):
            runtime.run_files(paths, out_path=out, async_writeback_mb=0)

        rates = [
            runtime.run_files(paths, out_path=out, async_writeback_mb=0).rows_per_second
            for _ in range(RUNS)
        ]
        median_mrows = median(rates) / 1e6
        print(f"case={CASE}")
        print(f"rows={ROWS:,} instruments={N} warmups={WARMUPS} runs={RUNS}")
        print(f"prefetch_rows={PREFETCH_ROWS}")
        print(f"inputs={runtime.program.input_names}")
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
