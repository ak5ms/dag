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
OUT = Path(os.environ.get("MPO_BENCH_OUTPUT_DIR", "/dev/shm/mpo_boyd_bench"))
MINUTE_US = 60_000_000.0
SESSION_ROWS = 1000
OPEN_ROWS = 980


def _fake_data(rows: int = ROWS, n_assets: int = N_ASSETS) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    base = 1_800_000_000_000_000.0
    row = np.arange(rows)
    ts1 = base + row * MINUTE_US
    session_id = row // SESSION_ROWS
    session_start1 = base + session_id * SESSION_ROWS * MINUTE_US
    session_end1 = session_start1 + OPEN_ROWS * MINUTE_US
    next_session_start1 = session_start1 + SESSION_ROWS * MINUTE_US
    next_session_end1 = next_session_start1 + OPEN_ROWS * MINUTE_US
    tradable1 = (ts1 >= session_start1) & (ts1 < session_end1)

    returns = rng.normal(scale=2e-4, size=(rows, n_assets))
    tradable = np.broadcast_to(tradable1[:, None], (rows, n_assets)).astype(float).copy()
    returns[tradable == 0.0] = 0.0
    for reopen in range(SESSION_ROWS, rows, SESSION_ROWS):
        returns[reopen] *= np.sqrt(SESSION_ROWS - OPEN_ROWS + 1.0)

    def lanes(x):
        return np.broadcast_to(x[:, None], (rows, n_assets)).astype(float).copy()

    return {
        "returns": returns,
        "is_tradable_out0": tradable,
        "vw_halfspread_out0": rng.uniform(3e-5, 8e-5, size=(rows, n_assets)),
        "_ev_ts": lanes(ts1),
        "session_start0": lanes(session_start1),
        "session_end0": lanes(session_end1),
        "next_session_start0": lanes(next_session_start1),
        "next_session_end0": lanes(next_session_end1),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = _fake_data()

    t0 = time.perf_counter()
    example._clarabel()
    clarabel_setup_s = time.perf_counter() - t0

    shutil.rmtree(example.CACHE / "clarabel", ignore_errors=True)
    t0 = time.perf_counter()
    runtime = compile_formula(
        list(example._formula(var("returns"))),
        data,
        n_instruments=N_ASSETS,
    )
    compile_s = time.perf_counter() - t0

    generated = runtime.generated_cpp.read_text()
    row_loop = "for (std::size_t t = row_begin; t < row_end; ++t)"
    assert generated.count(row_loop) == 1
    assert generated.count("stackdsl::ClarabelNode<") == 1

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
