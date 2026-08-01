from __future__ import annotations

import os
from pathlib import Path
from statistics import mean, median
import tempfile

import numpy as np

from trading_dsl_engine.cpp_stream import compile_npy_formula


ROWS = int(os.environ.get("CPP_STREAM_RIDGE_ROWS", "5000000"))
N = int(os.environ.get("CPP_STREAM_RIDGE_INSTRUMENTS", "9"))
RUNS = int(os.environ.get("CPP_STREAM_RIDGE_RUNS", "10"))
WARMUPS = int(os.environ.get("CPP_STREAM_RIDGE_WARMUPS", "1"))
CASE = os.environ.get("CPP_STREAM_RIDGE_CASE", "stateful_cat")
OUTPUT_DIR = os.environ.get("CPP_STREAM_BENCH_OUTPUT_DIR")
PREFETCH_ROWS = int(os.environ.get("CPP_STREAM_BENCH_PREFETCH_ROWS", "16"))
MIN_MROWS = float(os.environ.get("CPP_STREAM_RIDGE_MIN_MROWS", "0"))


def _write_npy(path: Path, seed: int, transform) -> None:
    rng = np.random.default_rng(seed)
    array = np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=np.float64,
        shape=(ROWS, N),
    )
    chunk = 131_072
    for start in range(0, ROWS, chunk):
        stop = min(start + chunk, ROWS)
        values = rng.normal(size=(stop - start, N))
        array[start:stop] = transform(values)
    array.flush()
    del array


def _build_data(root: Path) -> dict[str, Path]:
    paths = {name: root / f"{name}.npy" for name in ("x1", "x2", "x3", "y")}
    _write_npy(paths["x1"], 1, lambda x: x)
    _write_npy(paths["x2"], 2, lambda x: x)
    _write_npy(paths["x3"], 3, lambda x: x)

    x1 = np.load(paths["x1"], mmap_mode="r")
    x2 = np.load(paths["x2"], mmap_mode="r")
    x3 = np.load(paths["x3"], mmap_mode="r")
    y = np.lib.format.open_memmap(paths["y"], mode="w+", dtype=np.float64, shape=(ROWS, N))
    rng = np.random.default_rng(4)
    chunk = 131_072
    for start in range(0, ROWS, chunk):
        stop = min(start + chunk, ROWS)
        y[start:stop] = (
            0.4 * x1[start:stop]
            - 0.2 * x2[start:stop]
            + 0.1 * x3[start:stop]
            + rng.normal(scale=0.05, size=(stop - start, N))
        )
    y.flush()
    del y, x1, x2, x3
    return paths


def _formula() -> str:
    if CASE == "stateful_cat":
        return "get_preds(Ridge(cat(x1, x2, x3), y=y, hl=64, lambda_=0.1))"
    if CASE == "stateful_args":
        return "get_preds(Ridge(x1, x2, x3, y=y, hl=64, lambda_=0.1))"
    if CASE == "stateless_beta":
        return "get_beta(Ridge(cat(x1, x2, x3), y=y, hl=0, lambda_=0.1))"
    if CASE == "grouped_stateful":
        if N != 9:
            raise ValueError("grouped_stateful benchmark requires N=9")
        return (
            "groupby(univ([0], [1, 2], [3, 4, 5, 6, 7, 8]), x1, "
            "get_preds(Ridge(cat(self_, x2, x3), y=y, hl=64, lambda_=0.1)))"
        )
    if CASE == "cat_root":
        return "cat(x1, x2, x3)"
    raise ValueError(f"unknown CPP_STREAM_RIDGE_CASE={CASE!r}")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="cpp_stream_ridge_") as temporary:
        root = Path(temporary)
        paths = _build_data(root)
        formula = _formula()
        runtime = compile_npy_formula(
            formula,
            paths,
            n_instruments=N,
            prefetch_rows=PREFETCH_ROWS,
        )
        output_root = Path(OUTPUT_DIR) if OUTPUT_DIR else root
        output_root.mkdir(parents=True, exist_ok=True)
        output = output_root / f"cpp_stream_ridge_{CASE}.bin"

        for _ in range(WARMUPS):
            runtime.run_npy_files(paths, out_path=output, async_writeback_mb=0)
        rates = [
            runtime.run_npy_files(paths, out_path=output, async_writeback_mb=0).rows_per_second
            for _ in range(RUNS)
        ]
        median_mrows = median(rates) / 1e6
        values = np.memmap(
            output,
            mode="r",
            dtype=np.float64,
            shape=(ROWS, runtime.plan.output_row_width),
        )
        checksum = float(np.nansum(values[-min(1024, ROWS):]))
        print(f"case={CASE}")
        print(f"formula={formula}")
        print(f"rows={ROWS:,} instruments={N} warmups={WARMUPS} runs={RUNS}")
        print(f"output_row_width={runtime.plan.output_row_width}")
        print(f"scratch_slots={runtime.plan.scratch_slots}")
        print(f"median={median_mrows:.3f} M rows/s")
        print(f"mean={mean(rates) / 1e6:.3f} M rows/s")
        print(f"best={max(rates) / 1e6:.3f} M rows/s")
        print("runs=" + ", ".join(f"{value / 1e6:.3f}" for value in rates) + " M rows/s")
        print(f"checksum={checksum:.12g}")
        print(f"generated_cpp={runtime.generated_cpp}")
        if MIN_MROWS > 0 and median_mrows < MIN_MROWS:
            raise SystemExit(
                f"Ridge regression: median {median_mrows:.3f} M rows/s "
                f"is below CPP_STREAM_RIDGE_MIN_MROWS={MIN_MROWS:.3f}"
            )


if __name__ == "__main__":
    main()
