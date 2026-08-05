from __future__ import annotations

import os
import tempfile
from pathlib import Path
from statistics import mean, median

import numpy as np

from trading_dsl_engine.cpp_stream import compile_formula
from trading_dsl_engine.ir import compile_ir

ROWS = int(os.environ.get("CPP_STREAM_FUSION_ROWS", "5000000"))
N = int(os.environ.get("CPP_STREAM_FUSION_INSTRUMENTS", "9"))
RUNS = int(os.environ.get("CPP_STREAM_FUSION_RUNS", "10"))
WARMUPS = int(os.environ.get("CPP_STREAM_FUSION_WARMUPS", "1"))
PREFETCH_ROWS = int(os.environ.get("CPP_STREAM_FUSION_PREFETCH_ROWS", "16"))
THREADS = int(os.environ.get("CPP_STREAM_FUSION_THREADS", "1"))
VECTOR_WIDTH = int(os.environ.get("CPP_STREAM_FUSION_VECTOR_WIDTH", "16"))
CASE_TEXT = os.environ.get(
    "CPP_STREAM_FUSION_CASE",
    "ewm_co_skewness,ewm_co_kurtosis",
)
OUTPUT_DIR = os.environ.get("CPP_STREAM_BENCH_OUTPUT_DIR")


_FORMULAS = {
    "add_control": "x+y",
    "ewm_control": "ewm(x,span=32)",
    "ewm_co_skewness": "ewm_co_skewness(y,x,span=32)",
    "ewm_co_kurtosis": "ewm_co_kurtosis(y,x,span=32)",
    "ewm_co_skewness_compute": ('emit(ewm_co_skewness(y,x,span=32),mode="last")'),
    "ewm_co_kurtosis_compute": ('emit(ewm_co_kurtosis(y,x,span=32),mode="last")'),
    "ewm_co_skewness_pandas": (
        "ewm_co_skewness(y_missing,x_missing,span=32,min_periods=20,"
        "ignore_na=False,adjust=True)"
    ),
    "ewm_co_kurtosis_pandas": (
        "ewm_co_kurtosis(y_missing,x_missing,span=32,min_periods=20,"
        "ignore_na=False,adjust=True)"
    ),
    "ewm_partial_corr_pandas": (
        "ewm_partial_corr(x_missing,y_missing,z_missing,span=32,"
        "min_periods=20,ignore_na=False,adjust=True)"
    ),
    "vec_skewness": "vec_skewness(x_vec)",
    "vec_kurtosis": "vec_kurtosis(x_vec)",
    "rolling_median_256": "rolling_median(x,periods=256)",
    "rolling_median_16": "rolling_median(x,periods=16)",
    "rolling_median_64": "rolling_median(x,periods=64)",
    "rolling_rank_256": "rolling_pct_rank(x,periods=256)",
    "rolling_rank_16": "rolling_pct_rank(x,periods=16)",
    "rolling_rank_64": "rolling_pct_rank(x,periods=64)",
    "rolling_entropy_256": "rolling_entropy(x,periods=256,buckets=10)",
    "rolling_entropy_16": "rolling_entropy(x,periods=16,buckets=10)",
    "rolling_entropy_64": "rolling_entropy(x,periods=64,buckets=10)",
    "backfill_256": 'rolling_kth(x_missing,periods=256,k=1,ignore="NAN")',
    "prev_diff_256": "rolling_prev_diff(x_constant,periods=256)",
    "theilsen_257": "rolling_theilsen(y,x,periods=257,min_periods=257)",
    "theilsen_512": "rolling_theilsen(y,x,periods=512,min_periods=512)",
    "theilsen_513": "rolling_theilsen(y,x,periods=513,min_periods=513)",
    "ridge_metrics": (
        "cat(get_sse(Ridge(1,x,z,y=y,hl=32,lambda_=0.1)),"
        "get_sst(Ridge(1,x,z,y=y,hl=32,lambda_=0.1)),"
        "get_r2(Ridge(1,x,z,y=y,hl=32,lambda_=0.1)),"
        "get_residual_variance(Ridge(1,x,z,y=y,hl=32,lambda_=0.1)),"
        "get_effective_df(Ridge(1,x,z,y=y,hl=32,lambda_=0.1)),"
        "get_effective_n(Ridge(1,x,z,y=y,hl=32,lambda_=0.1)))"
    ),
    "ridge_metrics_compute": (
        "emit(cat(get_sse(Ridge(1,x,z,y=y,hl=32,lambda_=0.1)),"
        "get_sst(Ridge(1,x,z,y=y,hl=32,lambda_=0.1)),"
        "get_r2(Ridge(1,x,z,y=y,hl=32,lambda_=0.1)),"
        "get_residual_variance(Ridge(1,x,z,y=y,hl=32,lambda_=0.1)),"
        "get_effective_df(Ridge(1,x,z,y=y,hl=32,lambda_=0.1)),"
        "get_effective_n(Ridge(1,x,z,y=y,hl=32,lambda_=0.1))),"
        'mode="last")'
    ),
}


def _cases() -> tuple[str, ...]:
    cases = tuple(part.strip() for part in CASE_TEXT.split(",") if part.strip())
    unknown = sorted(set(cases) - set(_FORMULAS))
    if not cases or unknown:
        raise ValueError(
            f"invalid CPP_STREAM_FUSION_CASE={CASE_TEXT!r}; unknown={unknown}"
        )
    return cases


def _write_matrix(path: Path, seed: int, *, missing: bool = False) -> None:
    rng = np.random.default_rng(seed)
    values = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float64, shape=(ROWS, N)
    )
    chunk_rows = 131_072
    for start in range(0, ROWS, chunk_rows):
        stop = min(start + chunk_rows, ROWS)
        chunk = rng.normal(size=(stop - start, N))
        if missing:
            row_ids = np.arange(start, stop)[:, None]
            lane_ids = np.arange(N)[None, :]
            chunk[(row_ids + 17 * lane_ids + seed) % 127 == 0] = np.nan
        values[start:stop] = chunk
    values.flush()
    del values


def _build_data(root: Path, needed: set[str]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for name, seed, missing in (
        ("x", 1, False),
        ("y", 2, False),
        ("z", 3, False),
        ("x_missing", 4, True),
        ("y_missing", 5, True),
        ("z_missing", 6, True),
    ):
        if name not in needed:
            continue
        path = root / f"{name}.npy"
        _write_matrix(path, seed, missing=missing)
        paths[name] = path
    if "x_constant" in needed:
        constant = np.lib.format.open_memmap(
            root / "x_constant.npy",
            mode="w+",
            dtype=np.float64,
            shape=(ROWS, N),
        )
        constant[:] = 1.0
        constant.flush()
        del constant
        paths["x_constant"] = root / "x_constant.npy"
    if "x_vec" in needed:
        rng = np.random.default_rng(7)
        vector = np.lib.format.open_memmap(
            root / "x_vec.npy",
            mode="w+",
            dtype=np.float64,
            shape=(ROWS, N, VECTOR_WIDTH),
        )
        chunk_rows = max(1, 131_072 // VECTOR_WIDTH)
        for start in range(0, ROWS, chunk_rows):
            stop = min(start + chunk_rows, ROWS)
            vector[start:stop] = rng.normal(size=(stop - start, N, VECTOR_WIDTH))
        vector.flush()
        del vector
        paths["x_vec"] = root / "x_vec.npy"
    return paths


def _pin_process() -> tuple[int, ...]:
    if not hasattr(os, "sched_getaffinity"):
        return ()
    available = tuple(sorted(os.sched_getaffinity(0)))
    if not available:
        return ()
    requested = int(os.environ.get("CPP_STREAM_FUSION_CPU", available[0]))
    if requested not in available:
        raise ValueError(
            f"CPP_STREAM_FUSION_CPU={requested} is outside affinity {available}"
        )
    os.sched_setaffinity(0, {requested})
    return (requested,)


def _benchmark(
    case: str,
    paths: dict[str, Path],
    output_root: Path,
) -> dict[str, object]:
    formula = _FORMULAS[case]
    names = compile_ir(formula).input_names
    runtime = compile_formula(
        formula,
        {name: paths[name] for name in names},
        n_instruments=N,
        prefetch_rows=PREFETCH_ROWS,
    )
    output = output_root / f"cpp_stream_fusion_{case}.bin"
    for _ in range(WARMUPS):
        runtime.run(
            out_path=output,
            async_writeback_mb=0,
            threads=THREADS,
        )
    results = [
        runtime.run(
            out_path=output,
            async_writeback_mb=0,
            threads=THREADS,
        )
        for _ in range(RUNS)
    ]
    rates = [result.rows_per_second for result in results]
    values = np.memmap(
        output,
        mode="r",
        dtype=np.float64,
        shape=(
            1 if runtime.plan.output_mode == "final" else ROWS,
            runtime.plan.output_row_width,
        ),
    )
    tail = values[-min(1024, ROWS) :]
    checksum = float(np.nansum(tail))
    finite_fraction = float(np.count_nonzero(np.isfinite(tail)) / tail.size)
    del values
    output.unlink(missing_ok=True)
    return {
        "case": case,
        "formula": formula,
        "rates": rates,
        "seconds": [result.seconds for result in results],
        "median_mrows": median(rates) / 1e6,
        "mean_mrows": mean(rates) / 1e6,
        "best_mrows": max(rates) / 1e6,
        "checksum": checksum,
        "finite_fraction": finite_fraction,
        "stage_kinds": tuple(stage.kind for stage in runtime.plan.stages),
        "bundle_widths": tuple(
            len(getattr(stage, "bundle_outs", ()))
            for stage in runtime.plan.stages
            if getattr(stage, "bundle_outs", ())
        ),
        "scratch_slots": runtime.plan.scratch_slots,
        "generated_bytes": runtime.generated_cpp.stat().st_size,
        "generated_cpp": runtime.generated_cpp,
    }


def main() -> None:
    if ROWS <= 0 or N <= 0 or RUNS <= 0 or WARMUPS < 0 or THREADS < 0:
        raise ValueError(
            "rows, instruments, and runs must be positive; "
            "warmups and threads must be nonnegative"
        )
    affinity = _pin_process()
    cases = _cases()
    needed = {
        name for case in cases for name in compile_ir(_FORMULAS[case]).input_names
    }
    with tempfile.TemporaryDirectory(prefix="cpp_stream_fusion_") as temporary:
        root = Path(temporary)
        paths = _build_data(root, needed)
        output_root = Path(OUTPUT_DIR) if OUTPUT_DIR else root
        output_root.mkdir(parents=True, exist_ok=True)
        print(
            f"rows={ROWS:,} instruments={N} warmups={WARMUPS} runs={RUNS} "
            f"threads={THREADS} affinity={affinity}"
        )
        for case in cases:
            result = _benchmark(case, paths, output_root)
            print("---")
            print(f"case={case}")
            print(f"formula={result['formula']}")
            print(f"stage_kinds={result['stage_kinds']}")
            print(f"bundle_widths={result['bundle_widths']}")
            print(f"scratch_slots={result['scratch_slots']}")
            print(f"generated_bytes={result['generated_bytes']}")
            print(f"median={result['median_mrows']:.3f} M rows/s")
            print(f"mean={result['mean_mrows']:.3f} M rows/s")
            print(f"best={result['best_mrows']:.3f} M rows/s")
            print(
                "runs="
                + ", ".join(
                    f"{seconds:.6f}s/{rate / 1e6:.3f}M"
                    for seconds, rate in zip(
                        result["seconds"], result["rates"], strict=True
                    )
                )
            )
            print(f"checksum={result['checksum']:.12g}")
            print(f"finite_fraction={result['finite_fraction']:.6f}")
            print(f"generated_cpp={result['generated_cpp']}")


if __name__ == "__main__":
    main()
