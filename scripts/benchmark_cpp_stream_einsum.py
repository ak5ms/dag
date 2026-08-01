from __future__ import annotations

import os
from pathlib import Path
from statistics import mean, median
import tempfile

import numpy as np

from trading_dsl_engine.base.dsl import cat, einsum, var
from trading_dsl_engine.cpp_stream import compile_npy_formula


ROWS = int(os.environ.get("CPP_STREAM_EINSUM_ROWS", "5000000"))
N = int(os.environ.get("CPP_STREAM_EINSUM_INSTRUMENTS", "9"))
RUNS = int(os.environ.get("CPP_STREAM_EINSUM_RUNS", "10"))
WARMUPS = int(os.environ.get("CPP_STREAM_EINSUM_WARMUPS", "1"))
PREFETCH_ROWS = int(os.environ.get("CPP_STREAM_EINSUM_PREFETCH_ROWS", "16"))
CASE = os.environ.get("CPP_STREAM_EINSUM_CASE", "all")
OUTPUT_DIR = os.environ.get("CPP_STREAM_BENCH_OUTPUT_DIR")
MIN_MROWS = float(os.environ.get("CPP_STREAM_EINSUM_MIN_MROWS", "0"))


def _create(path: Path) -> np.memmap:
    return np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float64, shape=(ROWS, N)
    )


def _build_inputs(root: Path) -> dict[str, Path]:
    paths = {name: root / f"{name}.npy" for name in ("w", "x", "y", "z")}
    arrays = {name: _create(path) for name, path in paths.items()}
    lane = np.arange(N, dtype=np.float64)[None, :]
    chunk = 131_072
    for start in range(0, ROWS, chunk):
        stop = min(start + chunk, ROWS)
        t = np.arange(start, stop, dtype=np.float64)[:, None]
        arrays["w"][start:stop] = 0.25 + 1e-7 * t + 0.020 * lane
        arrays["x"][start:stop] = 1.00 + 2e-7 * t + 0.010 * lane
        arrays["y"][start:stop] = 2.00 - 1e-7 * t + 0.030 * lane
        arrays["z"][start:stop] = -0.50 + 3e-7 * t - 0.015 * lane
    for array in arrays.values():
        array.flush()
    arrays.clear()
    return paths


def _cases():
    x, y, z, w = var("x"), var("y"), var("z"), var("w")
    left6 = cat(x, y, z, w, x, y)
    right6 = cat(w, z, y, x, w, z)
    left2 = cat(x, y)
    middle2 = cat(y, z)
    right2 = cat(z, w)
    return {
        "row_dot": einsum("nf,nf->n", left6, right6),
        "ellipsis_dot": einsum("...f,...f->...", left6, right6),
        "scalar_reduce": einsum("n,n->", x, y),
        "nary_none": einsum(
            "ij,kj,kl->il", left2, middle2, right2, optimize=False
        ),
        "nary_greedy": einsum(
            "ij,kj,kl->il", left2, middle2, right2, optimize="greedy"
        ),
        "nary_optimal": einsum(
            "ij,kj,kl->il", left2, middle2, right2, optimize="optimal"
        ),
    }


def _selected_cases():
    cases = _cases()
    if CASE == "all":
        return cases
    requested = [item.strip() for item in CASE.split(",") if item.strip()]
    unknown = sorted(set(requested) - set(cases))
    if unknown:
        raise ValueError(f"unknown einsum benchmark case(s): {unknown}")
    return {name: cases[name] for name in requested}


def main() -> None:
    if ROWS <= 0 or N <= 0 or RUNS <= 0 or WARMUPS < 0:
        raise ValueError("rows/instruments/runs must be positive and warmups nonnegative")
    with tempfile.TemporaryDirectory(prefix="cpp_stream_einsum_") as temporary:
        root = Path(temporary)
        paths = _build_inputs(root)
        output_root = Path(OUTPUT_DIR) if OUTPUT_DIR else root
        output_root.mkdir(parents=True, exist_ok=True)

        for name, formula in _selected_cases().items():
            runtime = compile_npy_formula(
                formula,
                paths,
                n_instruments=N,
                prefetch_rows=PREFETCH_ROWS,
            )
            output = output_root / f"cpp_stream_einsum_{name}.bin"
            for _ in range(WARMUPS):
                runtime.run_npy_files(paths, out_path=output, async_writeback_mb=0)
            rates = [
                runtime.run_npy_files(
                    paths, out_path=output, async_writeback_mb=0
                ).rows_per_second
                for _ in range(RUNS)
            ]
            output_shape = (ROWS,) + tuple(runtime.plan.output_shape)
            values = np.memmap(
                output, mode="r", dtype=np.float64, shape=output_shape
            )
            tail = values[-min(8192, ROWS) :]
            checksum = float(np.nansum(tail))
            finite_fraction = float(np.isfinite(tail).mean())
            del values
            steps = [
                stage.einsum_step
                for stage in runtime.plan.stages
                if stage.kind == "einsum" and stage.einsum_step is not None
            ]
            estimated_flops = sum(step.estimated_flops for step in steps)
            median_mrows = median(rates) / 1e6
            print("---")
            print(f"case={name}")
            print(f"rows={ROWS:,} instruments={N} warmups={WARMUPS} runs={RUNS}")
            print(f"output_shape={runtime.plan.output_shape}")
            print(f"output_row_width={runtime.plan.output_row_width}")
            print(f"contraction_steps={len(steps)}")
            print(f"estimated_flops_per_row={estimated_flops}")
            print(f"largest_matrix_scratch_width={runtime.plan.matrix_scratch_width}")
            print(f"median={median_mrows:.6f} M rows/s")
            print(f"mean={mean(rates) / 1e6:.6f} M rows/s")
            print(f"best={max(rates) / 1e6:.6f} M rows/s")
            print(
                "runs="
                + ", ".join(f"{rate / 1e6:.6f}" for rate in rates)
                + " M rows/s"
            )
            print(f"checksum={checksum:.12g}")
            print(f"tail_finite_fraction={finite_fraction:.12g}")
            print(f"generated_cpp={runtime.generated_cpp}")
            if MIN_MROWS > 0 and median_mrows < MIN_MROWS:
                raise SystemExit(
                    f"{name} median {median_mrows:.6f} M rows/s is below "
                    f"CPP_STREAM_EINSUM_MIN_MROWS={MIN_MROWS:.6f}"
                )


if __name__ == "__main__":
    main()
