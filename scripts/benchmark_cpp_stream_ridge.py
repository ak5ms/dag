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
CASE_TEXT = os.environ.get("CPP_STREAM_RIDGE_CASE", "stateful_cat")
OUTPUT_DIR = os.environ.get("CPP_STREAM_BENCH_OUTPUT_DIR")
PREFETCH_ROWS = int(os.environ.get("CPP_STREAM_BENCH_PREFETCH_ROWS", "16"))
MIN_MROWS = float(os.environ.get("CPP_STREAM_RIDGE_MIN_MROWS", "0"))

_ALL_CASES = (
    "cat_root",
    "stateful_cat",
    "stateful_args",
    "stateless_beta",
    "grouped_one_stateful",
    "grouped_stateful",
)


def _cases() -> tuple[str, ...]:
    if CASE_TEXT == "all":
        return _ALL_CASES
    values = tuple(part.strip() for part in CASE_TEXT.split(",") if part.strip())
    unknown = sorted(set(values) - set(_ALL_CASES))
    if not values or unknown:
        raise ValueError(f"invalid CPP_STREAM_RIDGE_CASE={CASE_TEXT!r}; unknown={unknown}")
    return values


def _write_npy(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    array = np.lib.format.open_memmap(path, mode="w+", dtype=np.float64, shape=(ROWS, N))
    chunk = 131_072
    for start in range(0, ROWS, chunk):
        stop = min(start + chunk, ROWS)
        array[start:stop] = rng.normal(size=(stop - start, N))
    array.flush()
    del array


def _build_data(root: Path) -> dict[str, Path]:
    paths = {name: root / f"{name}.npy" for name in ("x1", "x2", "x3", "y")}
    _write_npy(paths["x1"], 1)
    _write_npy(paths["x2"], 2)
    _write_npy(paths["x3"], 3)

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


def _formula(case: str) -> str:
    if case == "stateful_cat":
        return "get_preds(Ridge(cat(x1, x2, x3), y=y, hl=64, lambda_=0.1))"
    if case == "stateful_args":
        return "get_preds(Ridge(x1, x2, x3, y=y, hl=64, lambda_=0.1))"
    if case == "stateless_beta":
        return "get_beta(Ridge(cat(x1, x2, x3), y=y, hl=0, lambda_=0.1))"
    if case == "grouped_one_stateful":
        if N != 9:
            raise ValueError("grouped_one_stateful benchmark requires N=9")
        return (
            "groupby(univ([0, 1, 2, 3, 4, 5, 6, 7, 8]), x1, "
            "get_preds(Ridge(cat(self_, x2, x3), y=y, hl=64, lambda_=0.1)))"
        )
    if case == "grouped_stateful":
        if N != 9:
            raise ValueError("grouped_stateful benchmark requires N=9")
        return (
            "groupby(univ([0], [1, 2], [3, 4, 5, 6, 7, 8]), x1, "
            "get_preds(Ridge(cat(self_, x2, x3), y=y, hl=64, lambda_=0.1)))"
        )
    if case == "cat_root":
        return "cat(x1, x2, x3)"
    raise AssertionError(case)


def _benchmark(case: str, paths: dict[str, Path], output_root: Path) -> dict[str, object]:
    formula = _formula(case)
    runtime = compile_npy_formula(formula, paths, n_instruments=N, prefetch_rows=PREFETCH_ROWS)
    output = output_root / f"cpp_stream_ridge_{case}.bin"
    for _ in range(WARMUPS):
        runtime.run_npy_files(paths, out_path=output, async_writeback_mb=0)
    rates = [
        runtime.run_npy_files(paths, out_path=output, async_writeback_mb=0).rows_per_second
        for _ in range(RUNS)
    ]
    values = np.memmap(
        output,
        mode="r",
        dtype=np.float64,
        shape=(ROWS, runtime.plan.output_row_width),
    )
    checksum = float(np.nansum(values[-min(1024, ROWS):]))
    del values
    output.unlink(missing_ok=True)
    return {
        "case": case,
        "formula": formula,
        "rates": rates,
        "median_mrows": median(rates) / 1e6,
        "mean_mrows": mean(rates) / 1e6,
        "best_mrows": max(rates) / 1e6,
        "checksum": checksum,
        "output_row_width": runtime.plan.output_row_width,
        "scratch_slots": runtime.plan.scratch_slots,
        "generated_cpp": runtime.generated_cpp,
    }


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="cpp_stream_ridge_") as temporary:
        root = Path(temporary)
        paths = _build_data(root)
        output_root = Path(OUTPUT_DIR) if OUTPUT_DIR else root
        output_root.mkdir(parents=True, exist_ok=True)
        print(f"rows={ROWS:,} instruments={N} warmups={WARMUPS} runs={RUNS}")
        print(f"prefetch_rows={PREFETCH_ROWS}")
        results = [_benchmark(case, paths, output_root) for case in _cases()]
        for result in results:
            print("---")
            print(f"case={result['case']}")
            print(f"formula={result['formula']}")
            print(f"output_row_width={result['output_row_width']}")
            print(f"scratch_slots={result['scratch_slots']}")
            print(f"median={result['median_mrows']:.3f} M rows/s")
            print(f"mean={result['mean_mrows']:.3f} M rows/s")
            print(f"best={result['best_mrows']:.3f} M rows/s")
            print("runs=" + ", ".join(f"{rate / 1e6:.3f}" for rate in result["rates"]) + " M rows/s")
            print(f"checksum={result['checksum']:.12g}")
            print(f"generated_cpp={result['generated_cpp']}")
            if MIN_MROWS > 0 and result["median_mrows"] < MIN_MROWS:
                raise SystemExit(
                    f"Ridge regression for {result['case']}: median "
                    f"{result['median_mrows']:.3f} M rows/s is below "
                    f"CPP_STREAM_RIDGE_MIN_MROWS={MIN_MROWS:.3f}"
                )


if __name__ == "__main__":
    main()
