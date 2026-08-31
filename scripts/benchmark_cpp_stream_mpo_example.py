from __future__ import annotations

import os
import shutil
import statistics
import time
from pathlib import Path

import numpy as np

from examples import cpp_stream_mpo_one_pass as example
from trading_dsl_engine.base.dsl import var
from trading_dsl_engine.cpp_stream import compile_formula

ROWS = int(os.environ.get("MPO_BENCH_ROWS", "5000"))
RUNS = int(os.environ.get("MPO_BENCH_RUNS", "5"))
N_ASSETS = 3
OUT = Path(os.environ.get("MPO_BENCH_OUTPUT_DIR", "/dev/shm/mpo_example_bench"))


def _fake_data(rows: int = ROWS, n_assets: int = N_ASSETS) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    returns = rng.normal(scale=2e-4, size=(rows, n_assets))
    tradable = np.ones((rows, n_assets), dtype=float)
    for start in range(1000, rows, 2000):
        stop = min(start + 10, rows)
        tradable[start:stop] = 0.0
        returns[start:stop] = 0.0
        if stop < rows:
            returns[stop] *= np.sqrt(stop - start + 1.0)
    hs = rng.uniform(3e-5, 8e-5, size=(rows, n_assets))
    ts = np.broadcast_to(
        (1_800_000_000_000_000 + np.arange(rows) * 60_000_000)[:, None],
        (rows, n_assets),
    ).copy()
    return {
        "returns": returns,
        "is_tradable_out0": tradable,
        "vw_halfspread_out0": hs,
        "_ev_ts": ts,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = _fake_data()

    # Native Clarabel is an install/cache dependency, not formula compilation.
    t0 = time.perf_counter()
    example._clarabel()
    clarabel_setup_s = time.perf_counter() - t0

    # Measure formula + CVXPY canonicalization/codegen + C++ compilation cold.
    shutil.rmtree(example.CACHE / "clarabel", ignore_errors=True)
    t0 = time.perf_counter()
    runtime = compile_formula(
        list(example._formula(var("returns"))),
        data,
        n_instruments=N_ASSETS,
    )
    compile_s = time.perf_counter() - t0

    warmup = runtime.run(out_path=OUT / "warmup.npy")
    print(f"rows={ROWS}")
    print(f"clarabel_setup_s={clarabel_setup_s:.6f}")
    print(f"compile_s={compile_s:.6f}")
    print(f"warmup_native_s={warmup.seconds:.6f}")

    wall_times = []
    native_times = []
    for i in range(RUNS):
        t0 = time.perf_counter()
        result = runtime.run(out_path=OUT / f"run_{i}.npy")
        wall = time.perf_counter() - t0
        wall_times.append(wall)
        native_times.append(result.seconds)
        print(
            f"run_{i + 1}: wall_s={wall:.6f} "
            f"wall_rows_per_s={ROWS / wall:.3f} "
            f"native_s={result.seconds:.6f} "
            f"native_rows_per_s={ROWS / result.seconds:.3f}"
        )

    mean_wall = statistics.mean(wall_times)
    median_wall = statistics.median(wall_times)
    mean_native = statistics.mean(native_times)
    median_native = statistics.median(native_times)
    print(f"mean_wall_s={mean_wall:.6f}")
    print(f"mean_wall_rows_per_s={ROWS / mean_wall:.3f}")
    print(f"median_wall_s={median_wall:.6f}")
    print(f"median_wall_rows_per_s={ROWS / median_wall:.3f}")
    print(f"mean_native_s={mean_native:.6f}")
    print(f"mean_native_rows_per_s={ROWS / mean_native:.3f}")
    print(f"median_native_s={median_native:.6f}")
    print(f"median_native_rows_per_s={ROWS / median_native:.3f}")


if __name__ == "__main__":
    main()
