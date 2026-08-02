from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np

from trading_dsl_engine.base.dsl import add, cumsum, ewm, mul, shift, var
from trading_dsl_engine.jax_flat.engine_cpp import compile_formula


DEFAULT_ROWS = 5_000_000
DEFAULT_COLS = 9


def _stateful_alpha():
    """Return an eight-stateful-level native-lowerable alpha DAG."""
    close = var("close")

    def native_ewm(x, span: float):
        return ewm(x, span, ignore_na=True, adjust=False)

    level1 = native_ewm(close, 8.0)
    level2 = cumsum(level1)
    level3 = native_ewm(level2, 16.0)
    level4 = shift(level3, 4, 16)
    level5 = native_ewm(add(level4, close), 32.0)
    level6 = cumsum(level5)
    level7 = native_ewm(add(level6, level3), 64.0)
    level8 = shift(level7, 8, 16)
    return add(mul(level8, level1), level5)


def _make_input(rows: int, cols: int, seed: int, path: Path | None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if path is None:
        return rng.normal(scale=0.001, size=(rows, cols)).astype(np.float64, copy=False)

    path.parent.mkdir(parents=True, exist_ok=True)
    close = np.memmap(path, mode="w+", dtype=np.float64, shape=(rows, cols))
    block_rows = max(1, min(rows, 250_000))
    for start in range(0, rows, block_rows):
        stop = min(start + block_rows, rows)
        close[start:stop] = rng.normal(scale=0.001, size=(stop - start, cols))
    close.flush()
    return close


def _checksum(out: np.ndarray) -> float:
    sample = np.asarray(out).reshape(-1)[::104_729]
    return float(np.sum(sample[np.isfinite(sample)], dtype=np.float64))


def _run_once(runtime, close: np.ndarray, out: np.ndarray) -> tuple[float, float]:
    state = runtime.init_state(close.shape[1])
    start = time.perf_counter()
    runtime.run_batch({"close": close}, states=state, out=out)
    elapsed = time.perf_counter() - start
    return elapsed, _checksum(out)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train or benchmark the native C++ runtime for GCC profile-guided optimization."
    )
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--cols", type=int, default=DEFAULT_COLS)
    parser.add_argument("--runs", type=int, default=6)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--input-memmap", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.rows <= 0 or args.cols <= 0 or args.runs <= 0 or args.warmups < 0:
        raise ValueError("rows, cols, and runs must be positive; warmups must be nonnegative")
    if args.cpu is not None:
        if not hasattr(os, "sched_setaffinity"):
            raise RuntimeError("--cpu requires os.sched_setaffinity support")
        os.sched_setaffinity(0, {args.cpu})

    close = _make_input(args.rows, args.cols, args.seed, args.input_memmap)
    out = np.empty((args.rows, args.cols), dtype=np.float64)
    runtime = compile_formula(_stateful_alpha())

    for _ in range(args.warmups):
        _run_once(runtime, close, out)

    times: list[float] = []
    checksums: list[float] = []
    for run in range(args.runs):
        elapsed, checksum = _run_once(runtime, close, out)
        times.append(elapsed)
        checksums.append(checksum)
        if not args.json:
            print(
                f"run={run + 1} seconds={elapsed:.9f} "
                f"rows_per_second={args.rows / elapsed:.3f} checksum={checksum:.17g}"
            )

    reference = checksums[0]
    if not all(np.isclose(value, reference, rtol=1e-12, atol=1e-12) for value in checksums[1:]):
        raise AssertionError(f"state-reset correctness failure: checksums={checksums}")

    result = {
        "rows": args.rows,
        "cols": args.cols,
        "stateful_depth": 8,
        "warmups": args.warmups,
        "runs": args.runs,
        "seconds": times,
        "best_seconds": min(times),
        "median_seconds": statistics.median(times),
        "median_rows_per_second": args.rows / statistics.median(times),
        "checksum": reference,
        "supported_ops": list(runtime.supported_ops),
    }
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(
            f"summary best_seconds={result['best_seconds']:.9f} "
            f"median_seconds={result['median_seconds']:.9f} "
            f"median_rows_per_second={result['median_rows_per_second']:.3f}"
        )


if __name__ == "__main__":
    main()
