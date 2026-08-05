from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
from statistics import median
import subprocess
import tempfile
from time import perf_counter

import numpy as np

from trading_dsl_engine.cpp_stream import compile_formula


ROOT = Path(__file__).resolve().parents[1]
NATIVE_SOURCE = ROOT / "scripts" / "cpp_stream_ewm_native_ceiling.cpp"
DEFAULT_CASES = (
    "copy",
    "ewm",
    "xs_rank_ewm",
    "co_skew",
    "co_kurt",
    "co_skew_general",
    "vec_skew",
    "vec_kurt",
    "ridge_metrics",
    "rolling_median_257",
    "backfill_257",
    "prev_diff_257",
)


def _scalar_data(rows: int, lanes: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(20260804)
    x = rng.normal(size=(rows, lanes))
    y = 0.45 * x + rng.normal(scale=0.85, size=(rows, lanes))
    weights = rng.uniform(0.2, 2.0, size=(rows, lanes))
    return {"x": x, "y": y, "weights": weights}


def _case(
    name: str,
    rows: int,
    lanes: int,
    vector_width: int,
) -> tuple[str, dict[str, np.ndarray], int]:
    if name in {"vec_skew", "vec_kurt"}:
        rng = np.random.default_rng(7821)
        values = rng.normal(size=(rows, lanes, vector_width))
        formula = {
            "vec_skew": "vec_skewness(v)",
            "vec_kurt": "vec_kurtosis(v)",
        }[name]
        return formula, {"v": values}, lanes
    if name in {"theilsen_257", "theilsen_513"}:
        periods = 257 if name.endswith("257") else 513
        actual_rows = min(rows, periods + 127)
        rng = np.random.default_rng(771)
        x = rng.normal(size=(actual_rows, 1))
        y = 2.25 * x + rng.normal(scale=0.3, size=(actual_rows, 1))
        return (
            f"rolling_theilsen(y, x, periods={periods}, "
            f"min_periods={periods})",
            {"x": x, "y": y},
            1,
        )

    data = _scalar_data(rows, lanes)
    x, y = data["x"], data["y"]
    if name == "prev_diff_runs_257":
        run = (np.arange(rows) // 512).astype(np.float64)
        data["x"] = np.broadcast_to(run[:, None], (rows, lanes)).copy()
    if name == "co_skew_general":
        x[::997, 0] = np.nan
        y[::733, min(1, lanes - 1)] = np.nan
    formulas = {
        "copy": "x",
        "ewm": "ewm(x, span=32)",
        "xs_rank_ewm": "xs_rank(ewm(x + 1, span=32))",
        "co_skew": "ewm_co_skewness(y, x, span=32)",
        "co_kurt": "ewm_co_kurtosis(y, x, span=32)",
        "co_skew_general": (
            "ewm_co_skewness(y, x, span=32, min_periods=10, "
            "ignore_na=False, adjust=True)"
        ),
        "rolling_median_257": (
            "rolling_median(x, periods=257, min_periods=64)"
        ),
        "backfill_257": (
            "rolling_kth(x, periods=257, k=1, ignore=\"NAN\", "
            "min_periods=1)"
        ),
        "prev_diff_257": "rolling_prev_diff(x, periods=257)",
        "prev_diff_runs_257": "rolling_prev_diff(x, periods=257)",
    }
    if name == "ridge_metrics":
        model = (
            "Ridge(1.0, x, y=y, weights=weights, hl=0, lambda_=0.2)"
        )
        formulas[name] = (
            "cat("
            f"get_sse({model}),get_sst({model}),get_r2({model}),"
            f"get_residual_variance({model}),get_effective_df({model}),"
            f"get_effective_n({model}))"
        )
    try:
        formula = formulas[name]
    except KeyError as exc:
        raise ValueError(f"unknown benchmark case {name!r}") from exc
    required = {
        key: value
        for key, value in data.items()
        if key in formula
    }
    return formula, required, lanes


def _benchmark_runtime(
    name: str,
    formula: str,
    data: dict[str, np.ndarray],
    lanes: int,
    output: Path,
    warmups: int,
    repeats: int,
) -> tuple[dict[str, object], object]:
    runtime = compile_formula(formula, data, n_instruments=lanes)
    for _ in range(warmups):
        runtime.run(out_path=output, threads=1)
    timings = [
        runtime.run(out_path=output, threads=1).seconds
        for _ in range(repeats)
    ]
    seconds = median(timings)
    rows = next(iter(data.values())).shape[0]
    return (
        {
            "case": name,
            "implementation": "cpp_stream",
            "rows": rows,
            "lanes": lanes,
            "seconds": seconds,
            "million_rows_per_second": rows / seconds / 1e6,
            "runs_seconds": timings,
            "stages": [stage.kind for stage in runtime.plan.stages],
            "bundle_members": [
                len(getattr(stage, "members", ()))
                for stage in runtime.plan.stages
                if getattr(stage, "members", ())
            ],
            "scratch_slots": runtime.plan.scratch_slots,
            "matrix_scratch_slots": runtime.plan.matrix_scratch_slots,
        },
        runtime,
    )


def _native_library(cache: Path) -> ctypes.CDLL:
    compiler = os.environ.get("CXX", "g++")
    digest = hashlib.sha256(NATIVE_SOURCE.read_bytes()).hexdigest()[:16]
    library = cache / f"ewm-native-ceiling-{digest}.so"
    if not library.is_file():
        cache.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                compiler,
                "-std=c++20",
                "-O3",
                "-march=native",
                "-mtune=native",
                "-fPIC",
                "-shared",
                str(NATIVE_SOURCE),
                "-o",
                str(library),
            ],
            check=True,
        )
    loaded = ctypes.CDLL(str(library))
    pointer = ctypes.POINTER(ctypes.c_double)
    common = [pointer, pointer, pointer, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_double]
    for name in (
        "ewm_co_skew_recursive_ceiling",
        "ewm_co_kurt_recursive_ceiling",
    ):
        getattr(loaded, name).argtypes = common
    loaded.ewm_co_skew_adjusted_ceiling.argtypes = [
        *common,
        ctypes.c_int64,
    ]
    return loaded


def _benchmark_native_ceiling(
    case: str,
    data: dict[str, np.ndarray],
    warmups: int,
    repeats: int,
    cache: Path,
) -> dict[str, object] | None:
    functions = {
        "co_skew": "ewm_co_skew_recursive_ceiling",
        "co_kurt": "ewm_co_kurt_recursive_ceiling",
        "co_skew_general": "ewm_co_skew_adjusted_ceiling",
    }
    if case not in functions:
        return None
    library = _native_library(cache)
    function = getattr(library, functions[case])
    x = np.ascontiguousarray(data["x"], dtype=np.float64)
    y = np.ascontiguousarray(data["y"], dtype=np.float64)
    cache.mkdir(parents=True, exist_ok=True)
    output = np.memmap(
        cache / f"native-output-{x.shape[0]}-{x.shape[1]}.bin",
        mode="w+",
        dtype=np.float64,
        shape=x.shape,
    )
    pointer = ctypes.POINTER(ctypes.c_double)
    arguments: list[object] = [
        x.ctypes.data_as(pointer),
        y.ctypes.data_as(pointer),
        output.ctypes.data_as(pointer),
        x.shape[0],
        x.shape[1],
        32.0,
    ]
    if case == "co_skew_general":
        arguments.append(10)
    for _ in range(warmups):
        function(*arguments)
    timings = []
    for _ in range(repeats):
        start = perf_counter()
        function(*arguments)
        timings.append(perf_counter() - start)
    seconds = median(timings)
    return {
        "case": case,
        "implementation": "specialized_native_ceiling",
        "rows": x.shape[0],
        "lanes": x.shape[1],
        "seconds": seconds,
        "million_rows_per_second": x.shape[0] / seconds / 1e6,
        "runs_seconds": timings,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=200_000)
    parser.add_argument("--lanes", type=int, default=9)
    parser.add_argument("--vector-width", type=int, default=16)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument(
        "--cases",
        default=",".join(DEFAULT_CASES),
        help=(
            "comma-separated cases; prev_diff_runs_257, theilsen_257, and "
            "theilsen_513 are available but not default"
        ),
    )
    parser.add_argument("--native-ceiling", action="store_true")
    parser.add_argument("--json", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="parent for the reused benchmark output (for example /dev/shm)",
    )
    args = parser.parse_args()
    if args.rows <= 0 or args.lanes <= 0 or args.repeats <= 0:
        parser.error("rows, lanes, and repeats must be positive")

    results: list[dict[str, object]] = []
    cache = Path(
        os.environ.get(
            "TRADING_DSL_ENGINE_CPP_STREAM_CACHE",
            tempfile.gettempdir(),
        )
    ) / "bench-native"
    with tempfile.TemporaryDirectory(
        prefix="cpp-stream-fusion-bench-",
        dir=args.output_dir,
    ) as tmp:
        root = Path(tmp)
        for name in (item.strip() for item in args.cases.split(",")):
            formula, data, lanes = _case(
                name, args.rows, args.lanes, args.vector_width
            )
            result, _ = _benchmark_runtime(
                name,
                formula,
                data,
                lanes,
                root / "output.bin",
                args.warmups,
                args.repeats,
            )
            results.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)
            if args.native_ceiling:
                ceiling = _benchmark_native_ceiling(
                    name, data, args.warmups, args.repeats, cache
                )
                if ceiling is not None:
                    results.append(ceiling)
                    print(json.dumps(ceiling, sort_keys=True), flush=True)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
